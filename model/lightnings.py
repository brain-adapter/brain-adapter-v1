import os
from typing import Dict, Union
from typing_extensions import override

import torch
import torchvision
import lightning
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch import nn
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    PNDMScheduler,
)
from transformers import CLIPTextModel, CLIPVisionModelWithProjection, CLIPTokenizer

from model.models import (
    EncoderModel,
    BrainVisionModel,
    AdapterModel,
    PreTrainedModel,
    VisionResamperModel,
    AdapterPipeline,
)
from model.modules import compute_snr, get_class


class LitBaseModel(lightning.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model: PreTrainedModel = None

    def forward(self, batch):
        raise NotImplementedError(
            "Cannot access this method here! Overwrite it instead."
        )

    def save_pretrained(self, save_directory: str):
        """
        Save the pure model states along with its config file
        """
        self.model.save_pretrained(save_directory=save_directory)

    def training_step(self, batch, batch_idx) -> Dict:
        model_outputs: Dict = self(batch)
        loss = model_outputs["loss"]

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        # log global steps
        self.log("step", self.global_step, on_step=True, prog_bar=False)

        return model_outputs

    def validation_step(self, batch, batch_idx) -> Dict:
        model_outputs = self(batch)
        loss = model_outputs["loss"]

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return model_outputs

    def test_step(self, batch, batch_idx) -> Dict:
        model_outputs = self(batch)
        loss = model_outputs["loss"]

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return model_outputs

    def configure_optimizers(self):
        config = self.config.optimizer

        params_to_optimize = [
            param for param in self.parameters() if param.requires_grad
        ]
        optimizer = get_class(config.name)(params_to_optimize, **config.params)

        config = self.config.get("scheduler", None)
        if config is not None:
            scheduler = get_class(config.name)(optimizer, **config.params)
        else:
            scheduler = None
        if scheduler is not None:
            return [optimizer], [scheduler]
        return [optimizer]


class LitBrainVisionModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = BrainVisionModel.from_pretrained(pretrained_model_path)
        else:
            self.model = BrainVisionModel(config=config.model)

        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        eeg_values, clip_embeds = (
            batch["eeg_values"],
            batch["clip_embeds"],
        )
        eeg_embeds, vision_embeds = self.model(eeg_values, clip_embeds)

        loss = 1 - nn.functional.cosine_similarity(
            eeg_embeds, vision_embeds, dim=-1
        ).mean(dim=-1)

        return {
            "eeg_embeds": eeg_embeds,
            "vision_embeds": vision_embeds,
            "loss": loss,
        }


class LitEEGClsModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = EncoderModel.from_pretrained(pretrained_model_path)
        else:
            self.model = EncoderModel(config=config.model)
        self.model.train()

    @override
    def forward(self, batch):
        eeg_values, labels, subjects = (
            batch["eeg_values"],
            batch["label"],
            batch["subject"],
        )
        logits: torch.Tensor = self.model(eeg_values, subjects)

        pred = logits.argmax(dim=-1)
        acc = torch.sum(pred == labels).cpu().item() / len(labels)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss, "acc": acc}

    @override
    def training_step(self, batch, batch_idx) -> Dict:
        model_outputs = super().training_step(batch, batch_idx)

        self.log("train_acc", model_outputs["acc"], on_epoch=True, on_step=False)

        return model_outputs

    @override
    def validation_step(self, batch, batch_idx) -> Dict:
        model_outputs = super().validation_step(batch, batch_idx)

        self.log("val_acc", model_outputs["acc"], on_epoch=True, on_step=False)

        return model_outputs

    @override
    def test_step(self, batch, batch_idx) -> Dict:
        model_outputs = super().test_step(batch, batch_idx)

        self.log("test_acc", model_outputs["acc"], on_epoch=True, on_step=False)

        return model_outputs


class LitAdapterModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.diffusion_model_path = config.lightning.diffusion_model_path
        self.snr_gamma = config.lightning.get("snr_gamma", None)

        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.diffusion_model_path, subfolder="unet"
        )

        # load adapter model
        unet_config = OmegaConf.create(dict(**self.unet.config))
        # load from pretrained model
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: AdapterModel = AdapterModel.from_pretrained(
                config.lightning.pretrained_model_path
            )
        # load from ip_adapter
        elif config.lightning.get("ip_adapter_model_path", None) is not None:
            OmegaConf.update(config.model, "unet_config", unet_config)
            self.model: AdapterModel = AdapterModel.from_ip_adapter(
                config.lightning.ip_adapter_model_path, config.model
            )
        # load from unet
        else:
            self.model: AdapterModel = AdapterModel.from_unet(self.unet, config.model)

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            self.diffusion_model_path, subfolder="vae"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.diffusion_model_path, subfolder="scheduler"
        )
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.diffusion_model_path, subfolder="text_encoder"
        )

        self.condition_encoder: Union[
            VisionResamperModel, EncoderModel, CLIPVisionModelWithProjection
        ] = get_class(config.lightning.condition_encoder.name).from_pretrained(
            config.lightning.condition_encoder.pretrained_model_path
        )

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.condition_encoder.requires_grad_(False)

        self.model.bind_unet(self.unet)
        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        pixel_values, condition_inputs, input_ids, drops = (
            batch["pixel_values"],
            batch["condition_inputs"],
            batch["input_ids"],
            batch["drops"],
        )
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents: torch.Tensor = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # get vision embeds
        with torch.no_grad():
            condition_embeds_without_drop = self.condition_encoder(condition_inputs)

        condition_embeds_ = []
        for condition_embed, drop in zip(condition_embeds_without_drop, drops):
            condition_embeds_.append(
                torch.where(drop, condition_embed, torch.zeros_like(condition_embed))
            )
        condition_embeds = torch.stack(condition_embeds_, dim=0)
        condition_embeds = self.model(condition_embeds)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(input_ids)[0]

        noise_pred: torch.Tensor = self.unet(
            noisy_latents,
            timesteps,
            torch.cat([prompt_embeds, condition_embeds], dim=1),
            return_dict=False,
        )[0]

        if self.snr_gamma is None:
            loss = nn.functional.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack(
                [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = nn.functional.mse_loss(
                noise_pred.float(), noise.float(), reduction="none"
            )
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return {"loss": loss}
    

    @override
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        pipeline = AdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        condition_inputs, ground_truth = (
            batch["condition_inputs"],
            batch["ground_truth"],
        )
        with torch.inference_mode():
            embeds = self.condition_encoder(condition_inputs)
            cond_embeds = self.model(embeds)
            uncond_embeds = self.model(torch.zeros_like(embeds))

        images: torch.Tensor = pipeline.generate(
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            seed=self.config.trainer.get("seed", 2024),
            output_type="pt",
        )
        images = images.cpu()
        ground_truth = ground_truth.cpu()

        image_grid = torchvision.utils.make_grid(
            torch.cat([ground_truth, images], dim=0), nrow=images.shape[0], padding=4
        )

        # log images to tensorboard logger
        self.logger.experiment.add_image(
            f"image_grid-{batch_idx}", image_grid, self.global_step, dataformats="CHW"
        )

    @override
    def test_step(self, batch, batch_idx):
        pipeline = AdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        condition_inputs, image_indexes = (
            batch["condition_inputs"],
            batch["image_indexes"],
        )
        with torch.inference_mode():
            embeds = self.condition_encoder(condition_inputs)
            cond_embeds = self.model(embeds)
            uncond_embeds = self.model(torch.zeros_like(embeds))

        # parameters for generation process
        num_images_per_prompt = self.config.trainer.test.get(
            "num_images_per_prompt", None
        )
        seed = self.config.trainer.test.get("seed", None)
        guidance_scale = self.config.trainer.test.get("guidance_scale", None)
        num_inference_steps = self.config.trainer.test.get("num_inference_steps", None)

        images: torch.Tensor = pipeline.generate(
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        log_directory = self.logger.log_dir
        if log_directory is not None:
            save_directory = os.path.join(log_directory, "images")
            # save generated images
            for i in range(len(condition_inputs)):
                for j in range(num_images_per_prompt):
                    image: Image = images[i * num_images_per_prompt + j]
                    save_path = os.path.join(
                        save_directory, f"{image_indexes[i]}_{j}.png"
                    )
                    image.save(save_path)
