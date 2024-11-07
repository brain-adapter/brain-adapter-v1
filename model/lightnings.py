import os
import io
from typing import Dict, Optional, List
from typing_extensions import override

import torch
import torchvision
import lightning
import matplotlib.pyplot as plt
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
from transformers import CLIPVisionModelWithProjection, CLIPTextModel


from model.activations import compute_snr, get_class
from model.models import (
    TransformerEncoderModel,
    VisionAdapterModel,
    BrainIPAdapterModel,
    BrainAdapterModel,
    MultiAdapterModel,
    PreTrainedModel,
    VisionAdapterPipeline,
    BrainAdapterPipeline,
    BrainIPAdapterPipeline,
    MultiAdapterPipeline,
)


class LitBaseModel(lightning.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.log_keys: List[str] = config.lightning.get("log_keys", [])
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
        loss = model_outputs.pop("loss")

        model_outputs = {
            f"train_{key}": model_outputs[key]
            for key in model_outputs.keys()
            if key in self.log_keys
        }

        # log global steps
        self.log("step", self.global_step, on_step=True, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if len(model_outputs) > 0:
            self.log_dict(model_outputs, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx) -> Dict:
        model_outputs: Dict = self(batch)
        loss = model_outputs.pop("loss")

        model_outputs = {
            f"val_{key}": model_outputs[key]
            for key in model_outputs.keys()
            if key in self.log_keys
        }
        # log global steps
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        if len(model_outputs) > 0:
            self.log_dict(model_outputs, prog_bar=False, on_epoch=True)

        return loss

    @rank_zero_only
    def test_step(self, batch, batch_idx) -> Dict:
        model_outputs: Dict = self(batch)
        loss = model_outputs.pop("loss")

        model_outputs = {
            f"test_{key}": model_outputs[key]
            for key in model_outputs.keys()
            if key in self.log_keys
        }
        # log global steps
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        if len(model_outputs) > 0:
            self.log_dict(model_outputs, prog_bar=False, on_epoch=True)

        return loss

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


class LitBrainModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = TransformerEncoderModel.from_pretrained(pretrained_model_path)
        else:
            self.model = TransformerEncoderModel(config=config.model)

        self.classifier = nn.Linear(self.model.embed_dim, config.dataset.num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.cls_ratio = config.lightning.get("cls_ratio", 0.5)

        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        eeg_values, image_embeds, labels, subjects = (
            batch["eeg_values"],
            batch["image_embeds"],
            batch["labels"],
            batch["subjects"],
        )
        eeg_embeds: torch.Tensor = self.model(eeg_values, subjects=subjects - 1)

        logits: torch.Tensor = self.classifier(eeg_embeds)

        clip_loss = nn.functional.mse_loss(eeg_embeds, image_embeds)
        cls_loss = self.cross_entropy(logits, labels)

        loss = clip_loss + cls_loss * self.cls_ratio

        pred = logits.argmax(dim=-1).detach()
        acc = torch.sum(pred == labels).cpu().item() / len(labels)

        return {
            "acc": acc,
            "cls_loss": cls_loss,
            "clip_loss": clip_loss,
            "loss": loss,
            "eeg_embeds": eeg_embeds,
            "vision_embeds": image_embeds,
        }

    @rank_zero_only
    def validation_step(self, batch, batch_idx) -> Dict:
        if not self.config.lightning.get("log_clip_logits", False):
            return super().validation_step(batch, batch_idx)

        model_outputs: Dict = self(batch)
        loss, eeg_embeds, vision_embeds, labels, clip_loss, cls_loss, acc = (
            model_outputs["loss"],
            model_outputs["eeg_embeds"],
            model_outputs["vision_embeds"],
            batch["labels"],
            model_outputs["clip_loss"],
            model_outputs["cls_loss"],
            model_outputs["acc"],
        )

        batch_size = len(labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_clip_loss", clip_loss, prog_bar=False, on_epoch=True)
        self.log("val_cls_loss", cls_loss, prog_bar=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=False, on_epoch=True)

        eeg_embeds = eeg_embeds / eeg_embeds.norm(p=2, dim=-1, keepdim=True)
        vision_embeds = vision_embeds / vision_embeds.norm(p=2, dim=-1, keepdim=True)

        clip_logits = torch.matmul(eeg_embeds, vision_embeds.t())
        clip_logits = clip_logits.detach().cpu()

        plt.figure(figsize=(16, 16))
        plt.imshow(clip_logits, cmap="viridis")

        plt.xlabel("vision-classes")
        plt.xticks(range(batch_size), labels.detach().cpu().tolist())

        plt.ylabel("eeg-classes")
        plt.yticks(range(batch_size), labels.detach().cpu().tolist())

        plt.colorbar()

        buf = io.BytesIO()
        # remove space
        plt.tight_layout(pad=0.0)
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
        buf.seek(0)

        self.logger.experiment.add_image(
            f"clip-logits-{batch_idx}",
            plt.imread(buf),
            self.global_step,
            dataformats="HWC",
        )
        plt.close()

        return loss


class LitBrainClsModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        pretrained_model_path = config.lightning.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            self.model = TransformerEncoderModel.from_pretrained(pretrained_model_path)
        else:
            self.model = TransformerEncoderModel(config=config.model)
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


class LitDiffusionModel(LitBaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.diffusion_model_path: str = config.lightning.diffusion_model_path
        self.snr_gamma: float = config.lightning.get("snr_gamma", None)

        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.diffusion_model_path, subfolder="unet"
        )

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            self.diffusion_model_path, subfolder="vae"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.diffusion_model_path, subfolder="scheduler"
        )

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.model = None

    def forward_loss(
        self, pixel_values: torch.FloatTensor, prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
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

        noise_pred: torch.Tensor = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
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

        return loss

    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("You should implement this method in the subclasses")

    @override
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        if not self.config.lightning.get("log_val_images", False):
            return None

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
        num_images_per_prompt = self.config.lightning.get("num_images_per_prompt", 1)
        seed = self.config.trainer.get("seed", None)
        num_inference_steps = self.config.lightning.get("num_inference_steps", 30)

        save_dir = self.logger.log_dir

        images: Image.Image = self.generate(
            batch, seed, num_inference_steps, num_images_per_prompt
        )
        image_indexes = batch["image_indexes"]
        batch_size = len(image_indexes)

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


class LitVisionAdapterModel(LitDiffusionModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.vision_model: CLIPVisionModelWithProjection = (
            CLIPVisionModelWithProjection.from_pretrained(
                config.lightning.clip_model_path
            )
        )

        # load from pretrained model
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: VisionAdapterModel = VisionAdapterModel.from_pretrained(
                config.lightning.pretrained_model_path, config.get("model", None)
            )
            self.model.bind_unet(self.unet)
        # load from unet
        else:
            self.model: VisionAdapterModel = VisionAdapterModel.from_unet(
                self.unet, config.model
            )

        self.vision_model.requires_grad_(False)
        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        pixel_values, clip_pixel_values, drops = (
            batch["pixel_values"],
            batch["clip_pixel_values"],
            batch["drops"],
        )

        with torch.no_grad():
            encoder_outputs = self.vision_model(clip_pixel_values).image_embeds

        cond_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(encoder_outputs, drops)
            ),
            dim=0,
        )
        cond_embeds: torch.FloatTensor = self.model(cond_embeds)

        loss = self.forward_loss(pixel_values, cond_embeds)

        return {"loss": loss}

    @override
    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        pipeline = VisionAdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        clip_pixel_values = batch["clip_pixel_values"]

        # get condition embeds
        with torch.inference_mode():
            cond_embeds = self.vision_model(clip_pixel_values).image_embeds
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


class LitBrainAdapterModel(LitDiffusionModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.brain_model = TransformerEncoderModel.from_pretrained(
            config.lightning.brain_model_path
        )

        # load from pretrained model, can be vision adapter or brain adapter
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: BrainAdapterModel = BrainAdapterModel.from_pretrained(
                config.lightning.pretrained_model_path, config.get("model", None)
            )
            self.model.bind_unet(self.unet)
        # load from unet
        else:
            self.model: BrainAdapterModel = BrainAdapterModel.from_unet(
                self.unet, config.model
            )

        self.brain_model.requires_grad_(False)
        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        pixel_values, eeg_values, subjects, drops = (
            batch["pixel_values"],
            batch["eeg_values"],
            batch["subjects"],
            batch["drops"],
        )

        with torch.no_grad():
            encoder_outputs = self.brain_model(eeg_values, subjects=subjects - 1)

        cond_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(encoder_outputs, drops)
            ),
            dim=0,
        )
        cond_embeds: torch.FloatTensor = self.model(cond_embeds)

        loss = self.forward_loss(pixel_values, cond_embeds)

        return {"loss": loss}

    @override
    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        pipeline = BrainAdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        eeg_values, subjects = (batch["eeg_values"], batch["subjects"])
        # get condition embeds
        with torch.inference_mode():
            cond_embeds = self.brain_model(eeg_values, subjects=subjects - 1)
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
    def test_step(self, batch, batch_idx):
        # parameters for generation process
        num_images_per_prompt = self.config.lightning.get("num_images_per_prompt", 1)
        seed = self.config.trainer.get("seed", None)
        num_inference_steps = self.config.lightning.get("num_inference_steps", 30)

        save_dir = self.logger.log_dir

        images: Image.Image = self.generate(
            batch, seed, num_inference_steps, num_images_per_prompt
        )
        image_indexes, subjects = (batch["image_indexes"], batch["subjects"])

        batch_size = len(image_indexes)

        if save_dir is not None:
            save_directory = os.path.join(save_dir, "images")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            # save generated images
            for i in range(batch_size):
                for j in range(num_images_per_prompt):
                    image: Image = images[i * num_images_per_prompt + j]
                    # Format:  {imageNet id} - {subject id} - {sample id} - {.png}
                    save_path = os.path.join(
                        save_directory, f"{image_indexes[i]}-{subjects[i]}-{j}.png"
                    )
                    image.save(save_path)


class LitBrainIPAdapterModel(LitBrainAdapterModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.brain_model: TransformerEncoderModel = (
            TransformerEncoderModel.from_pretrained(config.lightning.brain_model_path)
        )

        self.text_model: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.diffusion_model_path, subfolder="text_encoder"
        )

        # load from pretrained model, can be vision adapter or brain adapter
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: BrainIPAdapterModel = BrainIPAdapterModel.from_pretrained(
                config.lightning.pretrained_model_path, config.get("model", None)
            )
            self.model.bind_unet(self.unet)
        # load from ip-adapter
        else:
            self.model: BrainIPAdapterModel = BrainIPAdapterModel.from_ip_adapter(
                config.lightning.ip_adapter_model_path, config.model
            )
            self.model.bind_unet(self.unet)

        self.brain_model.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.model.train()

    @override
    def forward(self, batch) -> Dict:
        pixel_values, eeg_values, subjects, drops = (
            batch["pixel_values"],
            batch["eeg_values"],
            batch["subjects"],
            batch["drops"],
        )

        with torch.no_grad():
            encoder_outputs = self.brain_model(eeg_values, subjects=subjects - 1)

        brain_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(encoder_outputs, drops)
            ),
            dim=0,
        )
        brain_embeds: torch.FloatTensor = self.model(brain_embeds)

        with torch.no_grad():
            text_embeds = self.text_model(batch["text_input_ids"])[0]

        # no need to drop text prompt, since the prompt are fixed
        cond_embeds = torch.cat([text_embeds, brain_embeds], dim=1)

        loss = self.forward_loss(pixel_values, cond_embeds)

        return {"loss": loss}

    @override
    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        pipeline = BrainIPAdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                text_encoder=self.text_model,
                unet=self.unet,
                vae=self.vae,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        eeg_values, subjects = (
            batch["eeg_values"],
            batch["subjects"],
        )
        # get condition embeds
        with torch.inference_mode():
            brain_embeds = self.brain_model(eeg_values, subjects=subjects - 1)
        uncond_brain_embeds = torch.zeros_like(brain_embeds)

        with torch.inference_mode():
            brain_embeds = self.model(brain_embeds)
            uncond_brain_embeds = self.model(uncond_brain_embeds)

        prompt = ["best quality, high quality"] * len(eeg_values)
        negative_prompt = [
            "monochrome, lowres, bad anatomy, worst quality, low quality"
        ] * len(eeg_values)

        with torch.inference_mode():
            text_embeds, uncond_text_embeds = pipeline.pipeline.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        cond_embeds = torch.cat([text_embeds, brain_embeds], dim=1)
        uncond_embeds = torch.cat([uncond_text_embeds, uncond_brain_embeds], dim=1)

        images = pipeline(
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            seed=seed,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        return images


class LitMultiAdapterModel(LitDiffusionModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.vision_model: CLIPVisionModelWithProjection = (
            CLIPVisionModelWithProjection.from_pretrained(
                config.lightning.clip_model_path
            )
        )
        self.brain_model: TransformerEncoderModel = (
            TransformerEncoderModel.from_pretrained(config.lightning.brain_model_path)
        )

        # load from pretrained model
        if config.lightning.get("pretrained_model_path", None) is not None:
            self.model: MultiAdapterModel = MultiAdapterModel.from_pretrained(
                config.lightning.pretrained_model_path
            )
            self.model.bind_unet(self.unet)
        # load from vision_adapter
        else:
            self.model: MultiAdapterModel = MultiAdapterModel.from_vision_adapter(
                config.lightning.vision_adapter_model_path, config.model
            )

        self.model.train()
        self.vision_model.requires_grad_(False)
        self.brain_model.requires_grad_(False)

    @override
    def forward(self, batch) -> Dict:
        pixel_values = batch["pixel_values"]

        pixel_values, clip_pixel_values, eeg_values, subjects, drops = (
            batch["pixel_values"],
            batch["clip_pixel_values"],
            batch["eeg_values"],
            batch["subjects"],
            batch["drops"],
        )

        with torch.no_grad():
            vision_embeds = self.vision_model(clip_pixel_values).image_embeds
            brain_embeds = self.brain_model(eeg_values)

        vision_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(vision_embeds, drops)
            ),
            dim=0,
        )
        brain_embeds = torch.stack(
            tuple(
                torch.where(drop, torch.zeros_like(encoder_output), encoder_output)
                for encoder_output, drop in zip(brain_embeds, drops)
            ),
            dim=0,
        )

        cond_embeds: torch.FloatTensor = self.model(vision_embeds, brain_embeds)

        loss = self.forward_loss(pixel_values, cond_embeds)

        return {"loss": loss}

    def generate(
        self,
        batch,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        pipeline = MultiAdapterPipeline(
            StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_path,
                unet=self.unet,
                vae=self.vae,
                safety_checker=None,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        clip_pixel_values, eeg_values = (
            batch["clip_pixel_values"],
            batch["eeg_values"],
        )

        # get condition embeds
        with torch.inference_mode():
            vision_embeds = self.vision_encoder(clip_pixel_values).image_embeds
            brain_embeds = self.eeg_encoder(eeg_values)

        with torch.inference_mode():
            cond_embeds = self.model(vision_embeds, brain_embeds)
            uncond_embeds = self.model(
                torch.zeros_like(vision_embeds), torch.zeros_like(brain_embeds)
            )

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
    def test_step(self, batch, batch_idx):
        # parameters for generation process
        num_images_per_prompt = self.config.lightning.get("num_images_per_prompt", 1)
        seed = self.config.trainer.get("seed", None)
        num_inference_steps = self.config.lightning.get("num_inference_steps", 30)

        save_dir = self.logger.log_dir

        images: Image.Image = self.generate(
            batch, seed, num_inference_steps, num_images_per_prompt
        )
        image_indexes, subjects = (batch["image_indexes"], batch["subjects"])

        batch_size = len(image_indexes)

        if save_dir is not None:
            save_directory = os.path.join(save_dir, "images")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            # save generated images
            for i in range(batch_size):
                for j in range(num_images_per_prompt):
                    image: Image = images[i * num_images_per_prompt + j]
                    save_path = os.path.join(
                        save_directory, f"{image_indexes[i]}-{subjects[i]}-{j}.png"
                    )
                    image.save(save_path)

