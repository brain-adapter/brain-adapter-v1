import os
import copy
from typing import Optional, Tuple, Union, List, Dict
from typing_extensions import override

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from PIL import Image

from model.modules import (
    AttnProcessor,
    VisionAttnProcessor,
    AdapterProjection,
    EEGEmbeddings,
    get_class,
    get_generator,
    get_device,
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

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ):
        """
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.
        """
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

        model: nn.Module = cls(config)

        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

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

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor):
        encoder_outputs = self.encoder(inputs, output_attentions=False)

        embeds = encoder_outputs[0]

        return embeds


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
            config.vision_model_path
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.vision_model(
            pixel_values, output_hidden_states=False, return_dict=True
        ).image_embeds

        return image_embeds

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ) -> PreTrainedModel:
        if not isinstance(config, DictConfig) and pretrained_model_path is not None:
            conf_path = os.path.join(pretrained_model_path, "config.yml")
            if not os.path.exists(conf_path):
                raise FileNotFoundError(
                    f"Cannot find the config file in the model path: [{pretrained_model_path}]!"
                )
            config = OmegaConf.load(conf_path)
        else:
            assert config is not None
            # overwrite
            config = copy.deepcopy(config)

        model = cls(config)
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    def save_pretrained(self, save_directory: str):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        config_path = os.path.join(save_directory, "config.yml")
        # save the model config file as yaml
        OmegaConf.save(self.config, config_path)


class VisionModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_model_path)
        self.clip_skip = config.get("clip_skip", 1)
        self.post_norm = config.get("post_norm", True)

        self.post_layernorm = (
            nn.LayerNorm(
                self.vision_model.config.hidden_size,
                elementwise_affine=False,
                bias=False,
            )  # layernorm with no parameters
            if self.post_norm
            else nn.Identity()
        )

    def forward(self, pixel_values):
        model_outputs = self.vision_model(
            pixel_values, return_dict=True, output_hidden_states=True
        )

        embeds: torch.FloatTensor = model_outputs.hidden_states[-(self.clip_skip + 1)]

        pooled_output = embeds[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return pooled_output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ) -> PreTrainedModel:
        if not isinstance(config, DictConfig) and pretrained_model_path is not None:
            conf_path = os.path.join(pretrained_model_path, "config.yml")
            if not os.path.exists(conf_path):
                raise FileNotFoundError(
                    f"Cannot find the config file in the model path: [{pretrained_model_path}]!"
                )
            config = OmegaConf.load(conf_path)
        else:
            assert config is not None
            # overwrite
            config = copy.deepcopy(config)

        model = cls(config)
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    def save_pretrained(self, save_directory: str):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        config_path = os.path.join(save_directory, "config.yml")
        # save the model config file as yaml
        OmegaConf.save(self.config, config_path)


class AdapterModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.projection = AdapterProjection(config.projection_config)

        # Init using `from_unet` only
        unet = UNet2DConditionModel(**OmegaConf.to_object(config.unet_config))
        self.adapter_modules = nn.ModuleList(self._process_unet(unet).values())

        self.apply(self._init_weights)

    @staticmethod
    def _process_unet(
        unet: UNet2DConditionModel,
        load_weights: Optional[bool] = None,
    ) -> Dict:
        """
        Get attention processor dict of the given unet and replace processors
        """
        load_weights = load_weights if load_weights is not None else False

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
                attn_procs[name] = VisionAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
                if load_weights:
                    unet_sd = unet.state_dict()
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name].load_state_dict(weights)
        return attn_procs

    def forward(self, cond_embeds: torch.Tensor):
        return self.projection(cond_embeds)

    def bind_unet(self, unet: UNet2DConditionModel):
        unet.set_attn_processor(self._process_unet(unet))
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules.load_state_dict(self.adapter_modules.state_dict())

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        config: DictConfig,
        bind_unet: Optional[bool] = None,
    ) -> PreTrainedModel:
        unet_config = OmegaConf.create(dict(**unet.config))
        OmegaConf.update(config, "unet_config", unet_config)

        OmegaConf.update(
            config.projection_config,
            "cross_atttention_dim",
            unet.config.cross_attention_dim,
        )

        model = cls(config)

        attn_processor_dict = model._process_unet(unet, load_weights=True)

        adapter_modules = nn.ModuleList(attn_processor_dict.values())

        model.adapter_modules = adapter_modules

        bind_unet = bind_unet if bind_unet is not None else True
        if bind_unet:
            unet.set_attn_processor(attn_processor_dict)

        return model


class AdapterPipeline:
    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        condition_model: Optional[
            Union[VisionModel, EncoderModelWithProjection, VisionModelWithProjection]
        ] = None,
        adapter_model: Optional[AdapterModel] = None,
        processor: Optional[Union[EEGProcessor, CLIPImageProcessor]] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)
        self.dtype = dtype if dtype is not None else torch.float16

        self.pipeline = stable_diffusion_pipeline
        self.condition_model = condition_model
        self.adapter_model = adapter_model

        self.processor = processor

        if self.adapter_model is not None:
            self.adapter_model.bind_unet(self.pipeline.unet)

    def process_inputs(
        self, cond_inputs: Union[torch.Tensor, Image.Image, List[Image.Image]]
    ):
        if isinstance(self.processor, CLIPImageProcessor):
            return self.processor(cond_inputs, return_tensors="pt").pixel_values
        elif isinstance(self.processor, EEGProcessor):
            return self.processor(cond_inputs)
        else:
            raise ValueError(f"Invalid processor type [{type(self.processor)}]")

    @torch.inference_mode()
    def get_encoder_embeds(
        self, cond_inputs: Union[torch.Tensor, Image.Image, List[Image.Image]]
    ) -> Tuple[torch.Tensor]:
        assert cond_inputs is not None, "cond_inputs cannot be None"
        assert (
            self.processor is not None
        ), "Got condition inputs but the processor is None"

        cond_tensors: torch.Tensor = self.process_inputs(cond_inputs).to(
            self.device, self.dtype
        )
        assert (
            self.condition_model is not None
        ), "Got condition inputs but the condition model is None"
        embeds = self.condition_model(cond_tensors)

        assert (
            self.adapter_model is not None
        ), "Got condition inputs but the adapter model is None"
        cond_embeds = self.adapter_model(embeds)

        uncond_embeds = self.adapter_model(torch.zeros_like(embeds))

        return cond_embeds, uncond_embeds

    def generate(
        self,
        cond_inputs: Optional[
            Union[torch.Tensor, Image.Image, List[Image.Image]]
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
        if self.condition_model is not None and self.adapter_model is not None:
            self.condition_model.to(self.device, self.dtype)
            self.adapter_model.to(self.device, self.dtype)

        if cond_embeds is not None and uncond_embeds is not None:
            cond_embeds = cond_embeds.to(self.device, self.dtype)
            uncond_embeds = uncond_embeds.to(self.device, self.dtype)
        elif cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(cond_inputs)
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError("Got [cond_embeds], but [uncond_embeds] is missing")
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError("Got [uncond_embeds], but [cond_embeds] is missing")

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
