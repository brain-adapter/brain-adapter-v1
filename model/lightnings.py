import os
from typing import Dict, Optional, Union
from typing_extensions import override

import torch
import torchvision
import lightning
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from omegaconf import DictConfig
from torch import nn
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
)

from diffusers.image_processor import VaeImageProcessor

from model.models import (
    EncoderModel,
    EncoderModelWithProjection,
    VisionModelWithProjection,
    AdapterModel,
    PreTrainedModel,
    BlurReconstructionModel,
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

    @rank_zero_only
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

    @rank_zero_only
    def validation_step(self, batch, batch_idx) -> Dict:
        model_outputs = self(batch)
        loss = model_outputs["loss"]

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return model_outputs

    @rank_zero_only
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


class LitBrainKDModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = EncoderModelWithProjection.from_pretrained(
                pretrained_model_path
            )
        else:
            self.model = EncoderModelWithProjection(config=config.model)

        self.teacher_model = VisionModelWithProjection.from_pretrained(
            config.lightning.teacher_model_path
        )
        # self.adapter_model = AdapterModel.from_pretrained(
        #     config.lightning.adapter_model_path
        # )

        self.teacher_model.requires_grad_(False)
        # self.adapter_model.requires_grad_(False)

        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        eeg_values, pixel_values = (
            batch["eeg_values"],
            batch["pixel_values"],
        )
        eeg_embeds: torch.Tensor = self.model(eeg_values)

        # get teacher encoder embeds and teacher adapter tokens without gradient
        with torch.no_grad():
            teacher_embeds: torch.Tensor = self.teacher_model(pixel_values)
            # teacher_adapter_tokens = self.adapter_model(teacher_embeds)

        # eeg_adapter_tokens = self.adapter_model(eeg_embeds)

        normed_eeg_embeds = eeg_embeds / eeg_embeds.norm(p=2, dim=-1, keepdim=True)
        normed_teacher_embeds = teacher_embeds / teacher_embeds.norm(
            p=2, dim=-1, keepdim=True
        )

        encoder_loss = nn.functional.mse_loss(normed_eeg_embeds, normed_teacher_embeds)
        # adapter_loss = nn.functional.mse_loss(
        #     eeg_adapter_tokens, teacher_adapter_tokens
        # )

        loss = encoder_loss  # + adapter_loss

        return {
            # "adapter_loss": adapter_loss,
            "loss": loss,
        }


class LitEEGClsModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = EncoderModelWithProjection.from_pretrained(
                pretrained_model_path
            )
        else:
            self.model = EncoderModelWithProjection(config=config.model)
        self.model.train()

    @override
    def forward(self, batch):
        eeg_values, labels = (batch["eeg_values"], batch["label"])
        logits: torch.Tensor = self.model(eeg_values)

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
    @rank_zero_only
    def validation_step(self, batch, batch_idx) -> Dict:
        model_outputs = super().validation_step(batch, batch_idx)

        self.log("val_acc", model_outputs["acc"], on_epoch=True, on_step=False)

        return model_outputs

    @override
    @rank_zero_only
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

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            self.diffusion_model_path, subfolder="vae"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.diffusion_model_path, subfolder="scheduler"
        )

        self.condition_encoder: Union[
            VisionModelWithProjection, EncoderModelWithProjection
        ] = get_class(config.lightning.condition_model.name).from_pretrained(
            config.lightning.condition_model.model_path
        )

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.condition_encoder.requires_grad_(False)

        # load from pretrained model
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: AdapterModel = AdapterModel.from_pretrained(
                config.lightning.pretrained_model_path
            )
            self.model.bind_unet(self.unet)
        # load from unet
        else:
            self.model: AdapterModel = AdapterModel.from_unet(self.unet, config.model)

        self.model.train()
        if not config.lightning.get("projection_trainable", True):
            self.model.projection.requires_grad_(False)
            self.model.projection.eval()

    @override
    def forward(self, batch) -> Dict:
        pixel_values = batch["pixel_values"]
        # Encode pixel values into latent space
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

        conditions, drops = (batch["conditions"], batch["drops"])

        with torch.no_grad():
            encoder_outputs = self.condition_encoder(conditions)

        cond_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(encoder_outputs, drops)
            ),
            dim=0,
        )

        cond_embeds: torch.FloatTensor = self.model(cond_embeds)

        noise_pred: torch.Tensor = self.unet(
            noisy_latents,
            timesteps,
            cond_embeds,
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

    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        pipeline = AdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        conditions = batch["conditions"]

        # get condition embeds
        with torch.inference_mode():
            cond_embeds = self.condition_encoder(conditions)
        uncond_embeds = torch.zeros_like(cond_embeds)

        with torch.inference_mode():
            cond_embeds = self.model(cond_embeds)
            uncond_embeds = self.model(uncond_embeds)

        images = pipeline(
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        return images

    @override
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        seed = self.config.trainer.get("seed", None)
        num_inference_steps = self.config.lightning.get("num_inference_steps", 30)

        images: torch.Tensor = self.generate(
            batch, seed, num_inference_steps, output_type="pt"
        )
        images = images.cpu()
        ground_truth = batch["ground_truth"].cpu()

        image_grid = torchvision.utils.make_grid(
            torch.cat([ground_truth, images], dim=0), nrow=images.shape[0], padding=4
        )

        # log images to tensorboard logger
        self.logger.experiment.add_image(
            f"image_grid-{batch_idx}", image_grid, self.global_step, dataformats="CHW"
        )

    @override
    @rank_zero_only
    def test_step(self, batch, batch_idx):
        # parameters for generation process
        num_images_per_prompt = self.config.lightning.get("num_images_per_prompt", None)
        seed = self.config.trainer.get("seed", None)
        num_inference_steps = self.config.lightning.get("num_inference_steps", 30)

        batch_size = self.config.dataset.batch_size
        save_dir = self.logger.save_dir

        images: Image.Image = self.generate(
            batch, seed, num_inference_steps, num_images_per_prompt
        )
        image_indexes = batch["image_indexes"]

        if save_dir is not None:
            save_directory = os.path.join(save_dir, "images")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            # save generated images
            for i in range(batch_size):
                for j in range(num_images_per_prompt):
                    image: Image = images[i * num_images_per_prompt + j]
                    save_path = os.path.join(
                        save_directory, f"{image_indexes[i]}-{j}.png"
                    )
                    image.save(save_path)


class LitBlurReconModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = BlurReconstructionModel.from_pretrained(pretrained_model_path)
        else:
            self.model = BlurReconstructionModel(config=config.model)

        self.eeg_model = EncoderModel.from_pretrained(config.lightning.eeg_model_path)
        self.vae = AutoencoderKL.from_pretrained(
            config.lightning.diffusion_model_path, subfolder="vae"
        )

        self.eeg_model.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.config.scaling_factor
        )

        self.model.train()

    @override
    def forward(self, batch):
        eeg_values, pixel_values = (batch["eeg_values"], batch["pixel_values"])

        # pixel_values = kornia.filters.median_blur(pixel_values, (7, 7))

        with torch.no_grad():
            latents: torch.Tensor = self.vae.encode(pixel_values).latent_dist.sample()
            latents: torch.Tensor = latents * self.vae.config.scaling_factor

            eeg_embeds = self.eeg_model(eeg_values)

        latents_pred = self.model(eeg_embeds)

        loss = nn.functional.l1_loss(latents_pred, latents)

        return {"loss": loss, "latents_pred": latents_pred}

    @override
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        model_outputs = self(batch)

        loss, latents = (model_outputs["loss"], model_outputs["latents_pred"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        with torch.inference_mode():
            images = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]

        images = self.image_processor.postprocess(
            images, output_type="pt", do_denormalize=[True] * images.shape[0]
        )

        images = images.cpu()
        ground_truth = batch["ground_truth"].cpu()

        image_grid = torchvision.utils.make_grid(
            torch.cat([ground_truth, images], dim=0), nrow=images.shape[0], padding=4
        )
        # log images to tensorboard logger
        self.logger.experiment.add_image(
            f"image_grid-{batch_idx}", image_grid, self.global_step, dataformats="CHW"
        )

    @override
    @rank_zero_only
    def test_step(self, batch, batch_idx):
        pass
        # TODO implement this method if you need to do test process
