import os
import copy
from typing import Optional, Tuple, Union, List, Dict
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
        elif isinstance(module, EncoderModel):
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


class BrainVisionModel(PreTrainedModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.vision_model = EncoderModel(config.vision_config)
        self.eeg_model = EncoderModel(config.eeg_config)

    def forward(self, eeg_values: torch.Tensor, vision_hidden_states: torch.Tensor):
        eeg_embeds = self.eeg_model(eeg_values)

        vision_embeds = self.vision_model(vision_hidden_states)

        return eeg_embeds, vision_embeds

    def save_pretrained(self, save_directory: str):
        self.vision_model.save_pretrained(os.path.join(save_directory, "vision_model"))
        self.eeg_model.save_pretrained(os.path.join(save_directory, "eeg_model"))

        # save config
        config_path = os.path.join(save_directory, "config.yml")
        OmegaConf.save(self.config, config_path)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, config: Optional[DictConfig] = None
    ) -> PreTrainedModel:
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

        vision_model_path = os.path.join(pretrained_model_path, "vision-model")
        vision_model = EncoderModel.from_pretrained(vision_model_path)

        eeg_model_path = os.path.join(pretrained_model_path, "eeg-model")
        eeg_model = EncoderModel.from_pretrained(eeg_model_path)

        model = cls(config)
        model.vision_model.load_state_dict(vision_model.state_dict())
        model.eeg_model.load_state_dict(eeg_model.state_dict())

        return model


class VisionResamperModel(PreTrainedModel):
    """
    Wrapper for CLIP-ViT Model
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._clip_skip = config.get("clip_skip", 1)
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_model_path)
        self.resampler_model = EncoderModel.from_pretrained(config.resampler_model_path)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_model(
            pixel_values, output_hidden_states=True
        ).hidden_states[-(self._clip_skip + 1)]

        resample_features = self.resampler_model(hidden_states)

        return resample_features

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ):
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

        self.cross_attention_dim = config.get("cross_attention_dim", 768)
        self.input_size = config.get("input_size", 768)
        self.num_tokens = config.get("num_tokens", 4)

        self.projection = AdapterProjection(
            self.input_size, self.cross_attention_dim, self.num_tokens
        )

        unet = UNet2DConditionModel(**OmegaConf.to_object(config.unet_config))
        self.adapter_modules = nn.ModuleList(self._process_unet(unet).values())

        self.apply(self._init_weights)

    def _process_unet(self, unet: UNet2DConditionModel, num_tokens: int = 4) -> Dict:
        attn_procs = {}
        unet_sd = unet.state_dict()
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
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = VisionAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=num_tokens,
                )
                attn_procs[name].load_state_dict(weights, strict=False)

        return attn_procs

    def forward(self, eeg_embeds: torch.Tensor):
        return self.projection(eeg_embeds)

    def bind_unet(self, unet: UNet2DConditionModel):
        unet.set_attn_processor(
            self._process_unet(unet, num_tokens=self.num_tokens)
        )
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules.load_state_dict(self.adapter_modules.state_dict())

    @classmethod
    def from_unet(
        cls, unet: UNet2DConditionModel, config: DictConfig
    ) -> PreTrainedModel:
        unet_config = OmegaConf.create(dict(**unet.config))
        OmegaConf.update(config, "unet_config", unet_config)
        OmegaConf.update(config, "cross_attention_dim", unet.config.cross_attention_dim)

        model = cls(config)
        adapter_modules = nn.ModuleList(
            model._process_unet(unet, num_tokens=model.num_tokens).values()
        )
        model.adapter_modules.load_state_dict(adapter_modules.state_dict())
        return model

    @classmethod
    def from_ip_adapter(
        cls, ip_adapter_model_path: str, config: DictConfig
    ) -> PreTrainedModel:
        model = cls(config)
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(
            torch.stack([torch.sum(p) for p in model.projection.parameters()])
        )
        orig_adapter_sum = torch.sum(
            torch.stack([torch.sum(p) for p in model.adapter_modules.parameters()])
        )

        state_dict = torch.load(ip_adapter_model_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        model.projection.load_state_dict(state_dict["image_proj"], strict=True)
        model.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(
            torch.stack([torch.sum(p) for p in model.projection.parameters()])
        )
        new_adapter_sum = torch.sum(
            torch.stack([torch.sum(p) for p in model.adapter_modules.parameters()])
        )

        # Verify if the weights have changed
        assert (
            orig_ip_proj_sum != new_ip_proj_sum
        ), "Weights of image_proj_model did not change!"
        assert (
            orig_adapter_sum != new_adapter_sum
        ), "Weights of adapter_modules did not change!"

        return model


class AdapterPipeline:
    def __init__(
        self,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        condition_model: Optional[
            Union[VisionResamperModel, EncoderModel, CLIPVisionModelWithProjection]
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

        self.adapter_model.bind_unet(self.pipeline.unet)

    def set_scale(self, scale: float):
        for attn_processor in self.pipeline.unet.attn_processors.values():
            if isinstance(attn_processor, VisionAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_encoder_embeds(
        self, cond_inputs: Union[torch.Tensor, Image.Image, List[Image.Image]]
    ) -> Tuple[torch.Tensor]:
        assert cond_inputs is not None, "cond_inputs cannot be None"
        assert (
            self.processor is not None
        ), "Got condition inputs but the processor is None"

        cond_tensors: torch.Tensor = self.processor(cond_inputs, return_tensors="pt")
        assert (
            self.condition_model is not None
        ), "Got condition inputs but the condition model is None"
        embeds = self.condition_model(cond_tensors.to(self.device, self.dtype))

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
        prompts: Optional[Union[List[str], str]] = None,
        negative_prompts: Optional[Union[List, str]] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        self.pipeline.to(self.device, self.dtype)
        self.condition_model.to(self.device, self.dtype)
        self.adapter_model.to(self.device, self.dtype)

        if cond_embeds is None and uncond_embeds is None:
            cond_embeds, uncond_embeds = self.get_encoder_embeds(cond_inputs)
        elif cond_embeds is not None and uncond_embeds is None:
            raise ValueError(
                "Got [cond_embeds], but [uncond_embeds] is missing"
            )
        elif cond_embeds is None and uncond_embeds is not None:
            raise ValueError(
                "Got [uncond_embeds], but [cond_embeds] is missing"
            )            

        num_prompts = cond_embeds.shape[0]
        num_images_per_prompt = (
            num_images_per_prompt if num_images_per_prompt is not None else 1
        )

        if prompts is None:
            prompts = ""
        if negative_prompts is None:
            negative_prompts = ""

        if not isinstance(prompts, List):
            prompts = [prompts] * num_prompts
        if not isinstance(negative_prompts, List):
            negative_prompts = [negative_prompts] * num_prompts

        bs_embed, seq_len, _ = cond_embeds.shape
        cond_embeds = cond_embeds.repeat(1, num_images_per_prompt, 1)
        cond_embeds = cond_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        uncond_embeds = uncond_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_embeds = uncond_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipeline.encode_prompt(
                prompts,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
            )
            prompt_embeds = torch.cat([prompt_embeds_, cond_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds_, uncond_embeds], dim=1
            )

        generator = get_generator(seed, device=self.device)

        guidance_scale = guidance_scale if guidance_scale is not None else 7.5
        num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else 50
        )

        images = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
