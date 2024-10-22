import os
import copy
from typing import Optional, Tuple, Union, List, Dict, OrderedDict
from typing_extensions import override

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from torchvision.models import VisionTransformer
from PIL import Image

from model.activations import get_class, get_device, get_generator
from model.modules import (
    AttnProcessor,
    MixedAttnProcessor,
    AdapterProjection,
    EEGEmbeddings,
    EEGEmebeddingsWithMOE,
)
from data.dataset import EEGProcessor


class PreTrainedModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot access the base model!")

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, EEGEmbeddings):
            if module.class_embedding is not None:
                nn.init.normal_(
                    module.class_embedding, mean=0.0, std=module.embed_dim**-0.5
                )
        elif isinstance(module, EEGEmebeddingsWithMOE):
            for embedding in module.class_embedding:
                nn.init.normal_(embedding, mean=0.0, std=module.embed_dim**-0.5)

        elif isinstance(module, TransformerEncoderModel) and isinstance(
            module.projection, nn.Linear
        ):
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


class EncoderModel(PreTrainedModel):
    """
    Encoder model wrapper
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.encoder = get_class(config.encoder_name)(config)

        self.output_key: Union[int, str] = config.get("output_key", None)

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, **kwargs):
        encoder_outputs = self.encoder(inputs, **kwargs)

        if self.output_key is not None:
            encoder_outputs = encoder_outputs[self.output_key]

        return encoder_outputs


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


class TransformerEncoderModel(PreTrainedModel):
    """
    Encoder model wrapper with linear projection
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.encoder = get_class(config.encoder_name)(config)
        self.embed_dim = config.hidden_size

        self.projection = (
            nn.Linear(
                in_features=self.embed_dim,
                out_features=config.projection_dim,
                bias=False,
            )
            if config.get("projection_dim", None) is not None
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, values: torch.Tensor, **kwargs):
        encoder_outputs = self.encoder(values, output_attentions=False, **kwargs)

        embeds = self.projection(encoder_outputs[0])

        return embeds

    @torch.inference_mode()
    def get_attn_maps(self, values: torch.Tensor, **kwargs):
        encoder_outputs = self.encoder(values, output_attentions=True, **kwargs)

        attn_maps = encoder_outputs[2]

        return attn_maps


class PytorchVisionModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.vision_model = VisionTransformer(**config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)

    @override
    def save_pretrained(self, save_directory: str):
        raise ValueError("Cannot save an external pretrained model!")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ):
        config = cls.load_config(pretrained_model_path, config)

        model = cls(config)

        weights_path = os.path.join(pretrained_model_path, "pytorch_model.pth")
        weights = torch.load(weights_path)

        model.vision_model.load_state_dict(weights)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model


class AdapterModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        # init this model using method `from_unet`
        self.adapter_modules = None

    def _format_weights(self, to_key: torch.Tensor, to_value: torch.Tensor):
        return {
            "to_key.weight": copy.deepcopy(to_key),
            "to_value.weight": copy.deepcopy(to_value),
        }

    def _process_unet(
        self,
        unet: UNet2DConditionModel,
        load_weights: bool = False,
        processor_name: str = "model.modules.VisionAttnProcessor",
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
                processor: nn.Module = get_class(processor_name)(
                    hidden_size, cross_attention_dim
                )
                if load_weights:
                    unet_sd = unet.state_dict()
                    layer_name = name.split(".processor")[0]

                    weights = self._format_weights(
                        unet_sd[layer_name + ".to_k.weight"],
                        unet_sd[layer_name + ".to_v.weight"],
                    )
                    processor.load_state_dict(weights)

                attn_procs[name] = processor

        return attn_procs

    @classmethod
    def from_unet(
        cls, unet: UNet2DConditionModel, config: DictConfig, bind_unet: bool = True
    ):
        # update cross attention dim
        OmegaConf.update(
            config.projection_config,
            "cross_attention_dim",
            unet.config.cross_attention_dim,
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


class VisionAdapterModel(AdapterModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.projection = AdapterProjection(config.projection_config)

        self.apply(self._init_weights)

    def forward(self, cond_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(cond_embeds)


class BrainAdapterModel(AdapterModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.projection = AdapterProjection(config.projection_config)

        self.apply(self._init_weights)

    def forward(self, cond_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(cond_embeds)


class MultiAdapterModel(AdapterModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.vision_proj = AdapterProjection(config.vision_projection_config)
        self.eeg_proj = AdapterProjection(config.eeg_projection_config)

        self.apply(self._init_weights)

    def forward(self, vision_embeds: torch.Tensor, brain_embeds: torch.Tensor):
        vision_tokens = self.vision_proj(vision_embeds)
        brain_tokens = self.eeg_proj(brain_embeds)

        multi_tokens = torch.cat([vision_tokens, brain_tokens], dim=1)

        return multi_tokens

    @override
    def _format_weights(self, to_key: torch.Tensor, to_value: torch.Tensor):
        return {
            "to_key_eeg.weight": copy.deepcopy(to_key),
            "to_value_eeg.weight": copy.deepcopy(to_value),
            "to_key_vision.weight": copy.deepcopy(to_key),
            "to_value_vision.weight": copy.deepcopy(to_value),
            "eeg_scale": torch.tensor(0.0),
        }

    @override
    @classmethod
    def from_unet(
        cls, unet: UNet2DConditionModel, config: DictConfig, bind_unet: bool = True
    ):
        # update cross attention dim
        OmegaConf.update(
            config.vision_proj,
            "cross_attention_dim",
            unet.config.cross_attention_dim,
        )
        OmegaConf.update(
            config.eeg_proj,
            "cross_attention_dim",
            unet.config.cross_attention_dim,
        )

        model = cls(config)

        attn_processor_dict = model._process_unet(
            unet, load_weights=True, processor_name="model.modules.MixedAttnProcessor"
        )

        adapter_modules = nn.ModuleList(attn_processor_dict.values())

        model.adapter_modules = adapter_modules

        if bind_unet:
            unet.set_attn_processor(attn_processor_dict)

        return model

    @classmethod
    def from_vision_adapter(cls, vision_adapter_model_path: str, config: DictConfig):
        pretrained_config = MultiAdapterModel.load_config(vision_adapter_model_path)

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_config.diffusion_model_path, subfolder="unet"
        )

        model = cls.from_unet(unet, config, bind_unet=False)

        weights_path = os.path.join(vision_adapter_model_path, "pytorch_model.bin")
        weights: OrderedDict = torch.load(weights_path, map_location="cpu")

        # load proj weights
        proj_weights = {
            key.replace("projection.", ""): weights[key]
            for key in weights.keys()
            if key.startswith("projection.")
        }
        model.vision_proj.load_state_dict(proj_weights)
        model.eeg_proj.load_state_dict(proj_weights)

        # load adapter weights
        for index, module in enumerate(model.adapter_modules):
            if isinstance(module, MixedAttnProcessor):
                adapter_weights = {
                    key.replace(f"adapter_modules.{index}.", ""): weights[key]
                    for key in weights.keys()
                    if key.startswith("adapter_modules.")
                }

                module.load_state_dict(
                    model._format_weights(
                        adapter_weights["to_key.weight"],
                        adapter_weights["to_value.weight"],
                    )
                )

        model.eval()

        return model


def classifier_free_generate(
    diffusion_pipeline: StableDiffusionPipeline,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    num_images_per_prompt: Optional[int] = None,
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    **kwargs,
):
    cond_embeds = cond_embeds.to(diffusion_pipeline.device, diffusion_pipeline.dtype)
    uncond_embeds = uncond_embeds.to(
        diffusion_pipeline.device, diffusion_pipeline.dtype
    )

    num_images_per_prompt = (
        num_images_per_prompt if num_images_per_prompt is not None else 1
    )

    bs_embed, seq_len, _ = cond_embeds.shape

    cond_embeds = cond_embeds.repeat(1, num_images_per_prompt, 1)
    cond_embeds = cond_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    uncond_embeds = uncond_embeds.repeat(1, num_images_per_prompt, 1)
    uncond_embeds = uncond_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    generator = get_generator(seed, device=diffusion_pipeline.device)

    guidance_scale = guidance_scale if guidance_scale is not None else 7.5
    num_inference_steps = num_inference_steps if num_inference_steps is not None else 50

    images = diffusion_pipeline(
        prompt_embeds=cond_embeds,
        negative_prompt_embeds=uncond_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs,
    ).images

    return images


class AdapterPipeline:
    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.dtype = dtype if dtype is not None else torch.float16

        self.pipeline = stable_diffusion_pipeline


class VisionAdapterPipeline(AdapterPipeline):
    """
    Pipeline for vision adapter
    """

    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        vision_model: Optional[CLIPVisionModelWithProjection] = None,
        adapter_model: Optional[VisionAdapterModel] = None,
        processor: Optional[CLIPImageProcessor] = None,
    ):

        super().__init__(stable_diffusion_pipeline, device, dtype)

        self.vision_model = vision_model
        self.adapter_model = adapter_model
        self.processor = processor

        if self.adapter_model is not None:
            self.adapter_model.bind_unet(self.pipeline.unet)

    @torch.inference_mode()
    def get_encoder_embeds(
        self,
        pixel_values: Union[Image.Image, List[Image.Image]],
    ) -> Tuple[torch.Tensor]:
        assert pixel_values is not None, "pixel_values cannot be None"

        assert (
            self.processor is not None
        ), "Got condition inputs but the processor is None"

        assert (
            self.vision_model is not None
        ), "Got condition inputs but the condition model is None"

        assert (
            self.adapter_model is not None
        ), "Got condition inputs but the adapter model is None"

        cond_tensors = self.processor(pixel_values, return_tensors="pt").pixel_values

        cond_tensors = cond_tensors.to(self.device, self.dtype)

        cond_embeds = self.vision_model(cond_tensors).image_embeds
        uncond_embeds = torch.zeros_like(cond_embeds)

        cond_embeds = self.adapter_model(cond_embeds)
        uncond_embeds = self.adapter_model(uncond_embeds)

        return cond_embeds, uncond_embeds

    def __call__(
        self,
        pixel_values: Optional[Union[Image.Image, List[Image.Image]]] = None,
        cond_embeds: Optional[torch.Tensor] = None,
        uncond_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):

        self.pipeline.to(self.device, self.dtype)
        if self.vision_model is not None and self.adapter_model is not None:
            self.vision_model.to(self.device, self.dtype)
            self.adapter_model.to(self.device, self.dtype)

        if cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(pixel_values)
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError("Got [cond_embeds], but [uncond_embeds] is missing")
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError("Got [uncond_embeds], but [cond_embeds] is missing")

        return classifier_free_generate(
            self.pipeline,
            cond_embeds,
            uncond_embeds,
            num_images_per_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            **kwargs,
        )


class BrainAdapterPipeline(AdapterPipeline):
    """
    Pipeline for brain adapter
    """

    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        brain_model: Optional[TransformerEncoderModel] = None,
        adapter_model: Optional[BrainAdapterModel] = None,
        processor: Optional[EEGProcessor] = None,
    ):
        super().__init__(stable_diffusion_pipeline, device, dtype)

        self.brain_model = brain_model
        self.adapter_model = adapter_model
        self.processor = processor

        if self.adapter_model is not None:
            self.adapter_model.bind_unet(self.pipeline.unet)

    @torch.inference_mode()
    def get_encoder_embeds(
        self, eeg_values: torch.FloatTensor, subjects: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        assert (
            eeg_values is not None and subjects is not None
        ), "condition input cannot be None"

        assert (
            self.processor is not None
        ), "Got condition inputs but the processor is None"

        assert (
            self.brain_model is not None
        ), "Got condition inputs but the brain model is None"

        assert (
            self.adapter_model is not None
        ), "Got condition inputs but the adapter model is None"

        cond_tensors = self.processor(eeg_values)

        cond_tensors = cond_tensors.to(self.device, self.dtype)

        cond_embeds = self.condition_model(cond_tensors, subjects=subjects)
        uncond_embeds = torch.zeros_like(cond_embeds)

        cond_embeds = self.adapter_model(cond_embeds)
        uncond_embeds = self.adapter_model(uncond_embeds)

        return cond_embeds, uncond_embeds

    def __call__(
        self,
        eeg_values: Optional[torch.FloatTensor] = None,
        subjects: Optional[torch.Tensor] = None,
        cond_embeds: Optional[torch.Tensor] = None,
        uncond_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        self.pipeline.to(self.device, self.dtype)
        if self.brain_model is not None and self.adapter_model is not None:
            self.brain_model.to(self.device, self.dtype)
            self.adapter_model.to(self.device, self.dtype)

        if cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(eeg_values, subjects)
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError("Got [cond_embeds], but [uncond_embeds] is missing")
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError("Got [uncond_embeds], but [cond_embeds] is missing")

        return classifier_free_generate(
            self.pipeline,
            cond_embeds,
            uncond_embeds,
            num_images_per_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            **kwargs,
        )


class MultiAdapterPipeline(AdapterPipeline):
    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        brain_model: TransformerEncoderModel = None,
        vision_model: CLIPVisionModelWithProjection = None,
        adapter_model: Optional[MultiAdapterModel] = None,
        eeg_processor: EEGProcessor = None,
        vision_processor: CLIPImageProcessor = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(stable_diffusion_pipeline, device, dtype)

        self.brain_model = brain_model
        self.vision_model = vision_model
        self.adapter_model = adapter_model

        self.eeg_processor = eeg_processor
        self.vision_processor = vision_processor

        if self.adapter_model is not None:
            self.adapter_model.bind_unet(self.pipeline.unet)

    @torch.inference_mode()
    def get_encoder_embeds(
        self,
        eeg_values: torch.FloatTensor,
        subjects: torch.Tensor,
        pixel_values: Union[Image.Image, List[Image.Image]],
    ) -> Tuple[torch.Tensor]:
        assert (
            eeg_values is not None and pixel_values is not None
        ), "cond_inputs cannot be None"

        assert (
            self.eeg_processor is not None and self.vision_processor is not None
        ), "Got condition inputs but the processor is None"

        assert (
            self.brain_model is not None and self.vision_model is not None
        ), "Got condition inputs but the condition model is None"

        assert (
            self.adapter_model is not None
        ), "Got condition inputs but the adapter model is None"

        eeg_tensors = self.eeg_processor(eeg_values).to(self.device, self.dtype)
        vision_tensors = self.vision_processor(
            pixel_values, return_tensors="pt"
        ).pixel_values

        eeg_embeds = self.brain_model(eeg_tensors, subjects=subjects)
        vision_embeds = self.vision_model(vision_tensors)

        cond_embeds = torch.cat([vision_embeds, eeg_embeds], dim=1)
        uncond_embeds = torch.zeros_like(cond_embeds)

        cond_embeds = self.adapter_model(cond_embeds)
        uncond_embeds = self.adapter_model(uncond_embeds)

        return cond_embeds, uncond_embeds

    def __call__(
        self,
        eeg_values: torch.FloatTensor,
        pixel_values: Union[Image.Image, List[Image.Image]],
        cond_embeds: Optional[torch.Tensor] = None,
        uncond_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        self.pipeline.to(self.device, self.dtype)
        if (
            self.brain_model is not None
            and self.vision_model is not None
            and self.adapter_model is not None
        ):
            self.brain_model.to(self.device, self.dtype)
            self.adapter_model.to(self.device, self.dtype)
            self.vision_model.to(self.device, self.dtype)

        if cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(
                eeg_values, pixel_values
            )
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError("Got [cond_embeds], but [uncond_embeds] is missing")
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError("Got [uncond_embeds], but [cond_embeds] is missing")

        return classifier_free_generate(
            self.pipeline,
            cond_embeds,
            uncond_embeds,
            num_images_per_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            **kwargs,
        )
