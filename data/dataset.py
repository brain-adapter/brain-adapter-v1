import os
from typing import Dict, Optional, Union, List
import random
import json

import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer, ViTImageProcessor
from torchvision import transforms

from model.evaluation import resize_images
from model.modules import get_class


class EEGProcessor:
    def __init__(
        self,
        time_low: Optional[int],
        time_high: Optional[int],
        dtype: Optional[torch.dtype] = None,
    ):
        self._time_low = time_low if time_low is not None else 20
        self._time_high = time_high if time_high is not None else 460
        assert (
            self._time_low < self._time_high
        ), f"Invalid time_low[{time_low}] and time_high[{time_high}]"
        self.dtype = dtype if dtype is not None else torch.float32

    def __call__(
        self, eeg_values: torch.FloatTensor, return_batches: bool = True, **kwargs
    ) -> torch.Tensor:

        if eeg_values.ndim == 2:
            result = eeg_values[:, self._time_low : self._time_high]
            if return_batches:
                result = result.unsqueeze(dim=0)
        elif eeg_values.ndim == 3:  # already in batch form
            result = eeg_values[:, :, self._time_low : self._time_high]
        else:
            raise ValueError(f"Invalid ndim [{eeg_values.ndim}] of the eeg data")

        result = result.to(self.dtype)
        return result


class EEGImageNetDataset(Dataset):
    def __init__(self, mode: str, config: DictConfig):
        super().__init__()
        self.config = config
        self.mode = mode

        loaded = torch.load(config.eeg_data_path)
        dataset = loaded["dataset"]
        self.images = loaded["images"]

        # preprocess the raw data
        if "stddevs" in loaded.keys() and "means" in loaded.keys():
            stddevs = loaded["stddevs"]
            means = loaded["means"]
            for item in dataset:
                # do z-score for the raw eeg data
                item["eeg"] = (item["eeg"] - means) / stddevs

        # filter the dataset by subject id
        if config.subject != 0:
            self.dataset = [
                dataset[i]
                for i in range(len(dataset))
                if dataset[i]["subject"] == config.subject
            ]
        else:
            self.dataset = dataset

        # filter splitter
        splitter: list = torch.load(config.splitter_path)["splits"][0][mode]
        if (
            config.get("merge_train_and_val", False) and mode == "train"
        ):  # whether to merge the training and validation set
            splitter.extend(torch.load(config.splitter_path)["splits"][0]["val"])

        splitter = [i for i in splitter if i < len(self.dataset)]
        self.splitter = [
            i for i in splitter if 450 <= self.dataset[i]["eeg"].shape[-1] <= 600
        ]

        self.teacher_embeds = (
            torch.load(config.teacher_embeds_path)
            if config.get("teacher_embeds_path") is not None
            else None
        )

        self.eeg_processor = EEGProcessor(
            time_low=config.get("time_low", None),
            time_high=config.get("time_high", None),
        )

    def __len__(self):
        return len(self.splitter)

    def __getitem__(self, index) -> Dict:
        idx = self.splitter[index]
        item: Dict = self.dataset[idx]

        data = {
            "eeg_values": self.eeg_processor(item["eeg"], return_batches=False),
            "label": item["label"],
            "subject": item["subject"],
        }

        if self.teacher_embeds is not None:
            data["teacher_embeds"] = self.teacher_embeds[item["image"]]

        return data


class EEGImageNetDatasetForGeneration(EEGImageNetDataset):
    def __init__(self, mode: str, config: DictConfig) -> None:
        super().__init__(mode, config)

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(
                    config.resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(config.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_root_path = config.image_root_path
        self.resolution = config.get("resolution", 512)

        # handle drop to enhance classifier-free guidance
        self.text_drop_prob: float = config.get("text_drop_prob", 0.1)
        self.image_drop_prob: float = config.get("image_drop_prob", 0.05)

        if mode == "val" or self.mode == "test":
            self.splitter = random.sample(
                self.splitter, config.get("num_validation_images", 4)
            )

    def __getitem__(self, index) -> Dict:
        idx = self.splitter[index]
        item: Dict = self.dataset[idx]

        image_name: str = self.images[item["image"]]
        image_path = os.path.join(
            self.image_root_path,
            image_name.split("_")[0],
            ".".join([image_name, self.config.image_ext]),
        )
        raw_image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(raw_image)
        ground_truth = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        drop = torch.rand(2) < torch.tensor([self.image_drop_prob, self.text_drop_prob])

        eeg_inputs = self.eeg_processor(item["eeg"], return_batches=False)

        data = {
            "pixel_values": pixel_values,
            "image_indexes": item["image"],
            "vision_condition": eeg_inputs,
            "text_condition": eeg_inputs,
            "vision_drop": drop[0],
            "text_drop": drop[1],
        }

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = self.images[item["image"]]

        return data


class ImageDataset(Dataset):
    def __init__(self, mode: str, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = mode

        self.resolution = config.resolution
        self.image_root_path = config.image_root_path[mode]

        with open(config.meta_files[mode], "r") as f:
            self.meta: List = json.load(f)

        self.vae_processor = (
            transforms.Compose(
                [
                    transforms.Resize(
                        config.resolution,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            if mode == "train"
            else None
        )

        self.cond_processors = [
            CLIPImageProcessor.from_pretrained(model_path)
            for model_path in config.condition_model_paths
        ]

        self.drop_probability = config.drop_probability

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Dict:
        image_name: str = self.meta[index]["image"]
        image_path = os.path.join(self.image_root_path, image_name)
        raw_image = Image.open(image_path).convert("RGB")

        # for vae encoder
        pixel_values = self.vae_processor(raw_image) if self.mode == "train" else None

        # for inference
        ground_truth: Union[torch.FloatTensor, None] = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        # for vision condition
        conditions: List[torch.FloatTensor] = [
            processor(images=raw_image, return_tensors="pt").pixel_values
            for processor in self.cond_processors
        ]

        # handle drop to enhance classifier-free guidance
        drops = torch.rand(1) < self.drop_probability if self.mode == "train" else None

        data = {
            f"condition_{i}": conditions[i].squeeze(dim=0)
            for i in range(len(conditions))
        }

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = index

        elif self.mode == "train":
            data["pixel_values"] = pixel_values
            data["drops"] = drops

        return data


class ImageTextDataset(Dataset):
    def __init__(self, mode: str, config: DictConfig):
        super().__init__()
        self.config = config

        self.image_root_path = config.image_root_path[mode]
        self.mode = mode
        self.resolution = config.resolution

        with open(config.meta_files[mode], "r") as f:
            self.meta = json.load(f)

        self.vae_processor = (
            transforms.Compose(
                [
                    transforms.Resize(
                        config.resolution,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            if mode == "train"
            else None
        )

        self.vis_cond_processor = CLIPImageProcessor.from_pretrained(
            config.condition_model_path
        )

        self.text_cond_processor: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            config.condition_model_path
        )

        # handle drop to enhance classifier-free guidance
        self.drop_probability = config.drop_probability

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Dict:
        image_name: str = self.meta[index]["image"]
        image_path = os.path.join(self.image_root_path, image_name)
        raw_image = Image.open(image_path).convert("RGB")

        # for vae encoder
        pixel_values = self.vae_processor(raw_image) if self.mode == "train" else None

        # for inference
        ground_truth: Union[torch.FloatTensor, None] = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        # for vision adapter
        vision_condition: torch.FloatTensor = self.vis_cond_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values

        # for text adapter
        text = self.meta[index]["caption"]
        text_condition: torch.Tensor = self.text_cond_processor(
            text,
            max_length=self.text_cond_processor.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # handle drop to enhance classifier-free guidance
        drops = torch.rand(1) < self.drop_probability if self.mode == "train" else None

        data = {
            "condition_0": vision_condition.squeeze(dim=0),
            "condition_1": text_condition.squeeze(dim=0),
        }

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = index

        elif self.mode == "train":
            data["pixel_values"] = pixel_values
            data["drops"] = drops

        return data
