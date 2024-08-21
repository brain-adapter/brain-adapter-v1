import os
import copy
from typing import Optional, Tuple, Union, List, Dict, Mapping
from typing_extensions import override

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import (
    CLIPImageProcessor,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
)
from PIL import Image

from model.modules import (
    AttnProcessor,
    MixedAttnProcessor,
    AdapterProjection,
    EEGEmbeddings,
    get_class,
    get_generator,
    get_device,
)
from data.dataset import EEGProcessor
from model.evaluation import pt_to_numpy, numpy_to_pil


class PreTrainedModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot access the base model!")

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, EEGEmbeddings):
            nn.init.normal_(
                module.class_embedding, mean=0.0, std=module.embed_dim**-0.5
            )
            nn.init.normal_(module.patch_embedding.weight, std=0.02)
        elif isinstance(module, EncoderModelWithProjection):
            nn.init.normal_(
                module.projection.weight,
                std=module.embed_dim**-0.5,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(self, save_directory: str):
        """
        Save the current model along with its config
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        config_path = os.path.join(save_directory, "config.yml")
        model_path = os.path.join(save_directory, "pytorch_model.bin")

        # save the model config file as yaml
        OmegaConf.save(self.config, config_path)

        model = self.cpu()

        # save the model state_dict
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def load_config(pretrained_model_path: str, config: Optional[DictConfig] = None):
        if not isinstance(config, DictConfig):
            conf_path = os.path.join(pretrained_model_path, "config.yml")
            if not os.path.exists(conf_path):
                raise FileNotFoundError(
                    f"Cannot find the config file in the model path: [{pretrained_model_path}]!"
                )
            config = OmegaConf.load(conf_path)
        else:
            # overwrite
            config = copy.deepcopy(config)

        return config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ):
        """
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.
        """
        config = cls.load_config(pretrained_model_path, config)

        model: nn.Module = cls(config)

        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model.load_state_dict(copy.deepcopy(torch.load(model_path, map_location="cpu")))

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    def update_config(self, config: DictConfig):
        self.config = config


class JointModel(PreTrainedModel):
    """
    Wrapper for multi-pretrained models
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.models: Mapping[str, PreTrainedModel] = nn.ModuleDict(
            {
                key: get_class(model_config.name)(model_config)
                for key, model_config in config.items()
            }
        )

    def save_pretrained(self, save_directory: str):
        model_indexes: Dict[str, str] = {}
        for key, model in self.models.items():
            sub_folder = os.path.join(save_directory, key)

            model.save_pretrained(sub_folder)
            model_indexes[key] = {
                "model_path": sub_folder,
                "model_name": self.config.models[key].name,
            }

        index_config = OmegaConf.create(model_indexes)
        index_path = os.path.join(save_directory, "model_index.yml")

        OmegaConf.save(index_config, index_path)

    @override
    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ):
        index_path = os.path.join(pretrained_model_path, "model_index.yml")
        index_config = OmegaConf.load(index_path)

        models: Dict[str, PreTrainedModel] = {}
        configs = {}
        for key, item in index_config.items():
            models[key] = get_class(item.model_name).from_pretrained(item.model_path)
            configs[key] = models[key].config

        joint_model = cls(OmegaConf.create(configs))
        models: nn.ModuleDict = nn.ModuleDict(models)
        joint_model.models.load_state_dict(models.state_dict())

        return joint_model


class EncoderModel(PreTrainedModel):
    """
    Encoder model wrapper
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.encoder = get_class(config.encoder_name)(config)

        self.output_key: Union[int, str] = config.get("output_key", 1)

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor):
        encoder_outputs = self.encoder(inputs, output_attentions=False)

        return encoder_outputs[self.output_key]

    @override
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ) -> PreTrainedModel:
        config = cls.load_config(pretrained_model_path, config)

        model: nn.Module = cls(config)
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")

        loaded_ckpt = copy.deepcopy(torch.load(model_path, map_location="cpu"))

        model_keys = list(model.state_dict().keys())

        model_ckpt = {
            key: value for key, value in loaded_ckpt.items() if key in model_keys
        }

        model.load_state_dict(model_ckpt)

        model.eval()

        return model


class ExternalModel(PreTrainedModel):
    """
    Wrapper for external models
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.model = get_class(config.model_name)(**config.params)

        self.apply(self._init_weights)

    def forward(self, **inputs) -> torch.Tensor:
        return self.model(**inputs)


class EncoderModelWithProjection(PreTrainedModel):
    """
    Encoder model wrapper with linear projection
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.encoder = get_class(config.encoder_name)(config)
        self.embed_dim = config.hidden_size

        self.projection = nn.Linear(
            in_features=self.embed_dim, out_features=config.projection_dim, bias=False
        )
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor):
        encoder_outputs = self.encoder(inputs, output_attentions=False)

        embeds = self.projection(encoder_outputs[0])

        return embeds

    @torch.inference_mode()
    def get_attn_maps(self, inputs: torch.Tensor):
        encoder_outputs = self.encoder(inputs, output_attentions=True)

        attn_maps = encoder_outputs[1]

        return attn_maps


class VisionModelWithProjection(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(
            config.model_path
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        image_embeds = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        return image_embeds

    @override
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ) -> PreTrainedModel:
        config = OmegaConf.create({"model_path": pretrained_model_path})

        model = cls(config)
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    @override
    def save_pretrained(self, save_directory: str):
        raise ValueError("Cannot save an external pretrained model!")


class TextModelWithProjection(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(config.model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        text_embeds = self.text_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        return text_embeds

    @override
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ) -> PreTrainedModel:
        config = OmegaConf.create({"model_path": pretrained_model_path})

        model = cls(config)
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    @override
    def save_pretrained(self, save_directory: str):
        raise ValueError("Cannot save an external pretrained model!")


class AdapterModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.projections = nn.ModuleList([])
        self.token_bounds = ()

        for proj_conf in config.projection_configs:
            self.projections.append(AdapterProjection(proj_conf))
            self.token_bounds = self.token_bounds + (tuple(proj_conf.token_bounds),)

        # init this model using method `from_unet`
        self.adapter_modules = None

        self.apply(self._init_weights)

    def _process_unet(
        self,
        unet: UNet2DConditionModel,
        load_weights: bool = False,
    ) -> Dict[str, nn.Module]:
        """
        Get attention processor dict of the given unet and replace processors
        """

        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = MixedAttnProcessor(
                    hidden_size, cross_attention_dim, token_bounds=self.token_bounds
                )
                if load_weights:
                    unet_sd = unet.state_dict()
                    layer_name = name.split(".processor")[0]

                    # format weights
                    weights: Dict[str, torch.Tensor] = {}
                    for i in range(len(self.projections)):
                        weights[f"to_key.{i}.weight"] = copy.deepcopy(
                            unet_sd[layer_name + ".to_k.weight"]
                        )
                        weights[f"to_value.{i}.weight"] = copy.deepcopy(
                            unet_sd[layer_name + ".to_v.weight"]
                        )

                    attn_procs[name].load_state_dict(weights)

        return attn_procs

    def forward(self, cond_embeds: Tuple[torch.Tensor]) -> torch.Tensor:
        adapter_outputs = []
        for embeds, proj in zip(cond_embeds, self.projections):
            adapter_outputs.append(proj(embeds))

        adapter_outputs = torch.cat(adapter_outputs, dim=1)
        return adapter_outputs

    @classmethod
    def from_unet(
        cls, unet: UNet2DConditionModel, config: DictConfig, bind_unet: bool = True
    ):
        # set cross attention dimension for projection configs
        for proj_conf in config.projection_configs:
            OmegaConf.update(
                proj_conf, "cross_attention_dim", unet.config.cross_attention_dim
            )

        model = cls(config)

        attn_processor_dict = model._process_unet(unet, load_weights=True)

        adapter_modules = nn.ModuleList(attn_processor_dict.values())

        model.adapter_modules = adapter_modules

        if bind_unet:
            unet.set_attn_processor(attn_processor_dict)

        return model

    def bind_unet(self, unet: UNet2DConditionModel):
        unet.set_attn_processor(self._process_unet(unet))

        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules.load_state_dict(self.adapter_modules.state_dict())

        self.adapter_modules = adapter_modules

    @override
    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ) -> PreTrainedModel:
        """
        Note that after loading from a pretrained file, this model is NOT bound with a unet.
        You need to bind it with the unet before using this model.
        """
        config = cls.load_config(pretrained_model_path, config)

        unet = UNet2DConditionModel.from_pretrained(
            config.diffusion_model_path, subfolder="unet"
        )

        model = cls.from_unet(unet, config, bind_unet=False)

        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model.load_state_dict(copy.deepcopy(torch.load(model_path, map_location="cpu")))

        model.eval()

        return model


class AdapterPipeline:
    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        condition_models: Optional[nn.ModuleList] = None,
        adapter_model: Optional[AdapterModel] = None,
        processors: Optional[
            Tuple[Union[EEGProcessor, CLIPImageProcessor, CLIPTokenizer]]
        ] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.dtype = dtype if dtype is not None else torch.float16

        self.pipeline = stable_diffusion_pipeline
        self.condition_models = condition_models
        self.adapter_model = adapter_model

        self.processors = processors

        if self.adapter_model is not None:
            self.adapter_model.bind_unet(self.pipeline.unet)

    def process_inputs(
        self,
        processor: Union[EEGProcessor, CLIPImageProcessor, CLIPTokenizer],
        cond_inputs: Union[
            torch.FloatTensor, Image.Image, List[Image.Image], str, List[str]
        ],
    ) -> torch.Tensor:
        processed_inputs = None
        if isinstance(processor, CLIPImageProcessor):
            assert isinstance(
                cond_inputs, (Image.Image, List[Image.Image])
            ), f"Expect (Image.Image, List[Image.Image]), got {type(cond_inputs)}"
            processed_inputs = processor(cond_inputs, return_tensors="pt").pixel_values

        elif isinstance(processor, CLIPTokenizer):
            assert isinstance(
                cond_inputs, (str, List[str])
            ), f"Expect (str, List[str]), got {type(cond_inputs)}"
            processed_inputs = processor(
                cond_inputs,
                padding="max_length",
                max_length=processor.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        elif isinstance(processor, EEGProcessor):
            assert isinstance(
                cond_inputs, torch.FloatTensor
            ), f"Expect (torch.FloatTensor), got {type(cond_inputs)}"
            processed_inputs = processor(cond_inputs)
        else:
            raise ValueError(f"Unrecognizable processor type {type(processor)}")

        return processed_inputs

    @torch.inference_mode()
    def get_encoder_embeds(
        self,
        cond_inputs: Tuple[
            Union[torch.FloatTensor, Image.Image, List[Image.Image], str, List[str]]
        ],
    ) -> Tuple[torch.Tensor]:
        assert cond_inputs is not None, "cond_inputs cannot be None"

        assert (
            self.processors is not None
        ), "Got condition inputs but the processor is None"

        assert (
            self.condition_models is not None
        ), "Got condition inputs but the condition model is None"

        assert (
            self.adapter_model is not None
        ), "Got condition inputs but the adapter model is None"

        cond_embeds = ()
        uncond_embeds = ()

        for condition, processor, cond_model in zip(
            cond_inputs, self.processors, self.condition_models
        ):
            cond_tensors: torch.Tensor = self.process_inputs(processor, condition)

            model_embeds = cond_model(cond_tensors)

            cond_embeds = cond_embeds + (model_embeds,)
            uncond_embeds = uncond_embeds + (torch.zeros_like(model_embeds))

        uncond_embeds = self.adapter_model(uncond_embeds)
        cond_embeds = self.adapter_model(cond_embeds)

        return cond_embeds, uncond_embeds

    def __call__(
        self,
        cond_inputs: Optional[
            Dict[
                str,
                Union[
                    torch.FloatTensor, Image.Image, List[Image.Image], str, List[str]
                ],
            ]
        ] = None,
        cond_embeds: Optional[torch.Tensor] = None,
        uncond_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        self.pipeline.to(self.device, self.dtype)
        if self.condition_models is not None and self.adapter_model is not None:
            self.condition_models.to(self.device, self.dtype)
            self.adapter_model.to(self.device, self.dtype)

        if cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(cond_inputs)
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError("Got [cond_embeds], but [uncond_embeds] is missing")
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError("Got [uncond_embeds], but [cond_embeds] is missing")

        cond_embeds = cond_embeds.to(self.device, self.dtype)
        uncond_embeds = uncond_embeds.to(self.device, self.dtype)

        num_images_per_prompt = (
            num_images_per_prompt if num_images_per_prompt is not None else 1
        )

        bs_embed, seq_len, _ = cond_embeds.shape

        cond_embeds = cond_embeds.repeat(1, num_images_per_prompt, 1)
        cond_embeds = cond_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        uncond_embeds = uncond_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_embeds = uncond_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        generator = get_generator(seed, device=self.device)

        guidance_scale = guidance_scale if guidance_scale is not None else 7.5
        num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else 50
        )

        images = self.pipeline(
            prompt_embeds=cond_embeds,
            negative_prompt_embeds=uncond_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class BlurReconstructionModel(JointModel):
    """
    For blur reconstruction
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # perceiver resampler, EncoderModel
        resampler = self.models["resampler"]
        resampler_results: torch.Tensor = resampler(inputs)

        feature_width = int(self.config.resampler.query_tokens**0.5)

        decoder_inputs = resampler_results.transpose(1, 2).reshape(
            resampler_results.shape[0], -1, feature_width, feature_width
        ).contiguous()

        # vae decoder, ExternalModel
        # diffusers.models.autoencoders.vae.Decoder
        decoder = self.models["decoder"]
        decoder_results = decoder(sample=decoder_inputs)

        return decoder_results


class BlurReconstructionPipeline:
    def __init__(
        self,
        reconstruction_model: BlurReconstructionModel,
        vae: AutoencoderKL,
        eeg_model: Optional[EncoderModel] = None,
        processor: Optional[EEGProcessor] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.dtype = dtype if dtype is not None else torch.float16

        self.eeg_model = eeg_model
        self.vae = vae
        self.reconstruction_model = reconstruction_model

        self.processor = processor

    @torch.inference_mode()
    def get_eeg_embeds(self, eeg_values: torch.Tensor):
        assert self.processor is not None, "Got eeg values but the processor is None"
        eeg_inputs = self.processor(eeg_values)

        assert self.eeg_model is not None, "Got eeg values but the eeg_model is None"
        eeg_embeds = self.eeg_model(eeg_inputs)

        return eeg_embeds

    def __call__(
        self,
        eeg_values: Optional[torch.Tensor] = None,
        eeg_embeds: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        output_type: str = "pil",  # pil, pt, np
    ) -> Union[Image.Image, torch.Tensor]:
        self.reconstruction_model.to(self.device, self.dtype)
        self.vae.to(self.device, self.dtype)

        if self.eeg_model is not None:
            self.eeg_model.to(self.device, self.dtype)

        if eeg_embeds is None and eeg_values is not None:
            eeg_embeds = self.get_eeg_embeds(eeg_values)

        eeg_embeds = eeg_embeds.to(self.device, self.dtype)

        generator = get_generator(seed, device=self.device)

        with torch.inference_mode():
            latents_pred = self.reconstruction_model(eeg_embeds)
            pixel_pred = self.vae.decode(
                latents_pred / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]

        pixel_values = pixel_pred / 2.0 + 0.5

        results = pixel_values.clamp(0, 1)

        if output_type == "np":
            results = pt_to_numpy(results)
        if output_type == "pil":
            results = numpy_to_pil(results)

        return results
