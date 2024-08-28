import os
from typing import Dict, Optional, Union, List
import random
import json

import torch
import tarfile
from torch.utils.data import Dataset
from omegaconf import DictConfig
from PIL import Image, TarIO
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
        self.image_root_path: str = config.image_root_path
        self.image_ext: str = config.image_ext

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

        # processors for eeg and images

        self.eeg_processor = EEGProcessor(
            time_low=config.get("time_low", None),
            time_high=config.get("time_high", None),
        )

        self.image_processor = (
            CLIPImageProcessor.from_pretrained(config.teacher_model_path)
            if config.get("teacher_model_path", None) is not None
            else None
        )

    def __len__(self):
        return len(self.splitter)

    def __getitem__(self, index) -> Dict:
        idx = self.splitter[index]
        item: Dict = self.dataset[idx]

        image_path = os.path.join(
            self.image_root_path,
            ".".join([self.images[item["image"]], self.image_ext]),
        )
        raw_image = Image.open(image_path).convert("RGB")
        vision_inputs = self.image_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values.squeeze(dim=0)

        eeg_inputs = self.eeg_processor(item["eeg"], return_batches=False)

        data = {
            "eeg_values": eeg_inputs,
            "pixel_values": vision_inputs,
            "labels": item["label"],
            "subjects": item["subject"],
        }

        return data


class EEGImageNetDatasetForBlurReconstruction(EEGImageNetDataset):
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
        self.resolution: int = config.resolution

        if mode == "val" or self.mode == "test":
            self.splitter = random.sample(
                self.splitter, config.get("num_validation_images", 4)
            )

    def __getitem__(self, index) -> Dict:
        idx = self.splitter[index]
        item: Dict = self.dataset[idx]

        image_path = os.path.join(
            self.image_root_path,
            ".".join([self.images[item["image"]], self.image_ext]),
        )
        raw_image = Image.open(image_path).convert("RGB")

        vision_inputs = self.image_processor(raw_image)
        ground_truth = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        eeg_inputs = self.eeg_processor(item["eeg"], return_batches=False)

        data = {
            "pixel_values": vision_inputs.squeeze(dim=0),
            "eeg_values": eeg_inputs,
        }

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = self.images[item["image"]]

        return data


class EEGImageNetDatasetForGeneration(EEGImageNetDatasetForBlurReconstruction):
    def __init__(self, mode: str, config: DictConfig) -> None:
        super().__init__(mode, config)

        self.drop_probability: float = config.drop_probability

    def __getitem__(self, index) -> Dict:
        idx = self.splitter[index]
        item: Dict = self.dataset[idx]

        image_path = os.path.join(
            self.image_root_path,
            ".".join([self.images[item["image"]], self.image_ext]),
        )
        raw_image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values
        ground_truth = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        # handle drop to enhance classifier-free guidance
        drops = torch.rand(1) < self.drop_probability if self.mode == "train" else None

        eeg_inputs = self.eeg_processor(item["eeg"], return_batches=False)

        data = {
            "pixel_values": pixel_values.squeeze(dim=0),
            "image_indexes": item["image"],
            "conditions": eeg_inputs,
            "drops": drops,
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

        self.resolution: int = config.resolution
        self.image_root_path:str = config.image_root_path[mode]

        self.meta:List = [
            file
            for file in os.listdir(self.image_root_path)
            if file.endswith(config.image_ext)
        ]

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

        self.cond_processor = CLIPImageProcessor.from_pretrained(
            config.condition_model_path
        )

        self.drop_probability: float = config.drop_probability

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Dict:
        image_name: str = self.meta[index]
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
        conditions: torch.FloatTensor = self.cond_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values

        # handle drop to enhance classifier-free guidance
        drops = torch.rand(1) < self.drop_probability if self.mode == "train" else None

        data = {"conditions": conditions.squeeze(dim=0)}

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = index

        elif self.mode == "train":
            data["pixel_values"] = pixel_values
            data["drops"] = drops

        return data


class TarImageDataset(Dataset):
    """
    Dataset for images zipped in tarfiles
    """

    def __init__(self, mode: str, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = mode

        self.resolution: int = config.resolution
        self.image_root_path = config.image_root_path[mode]

        tarfiles = [
            os.path.join(self.image_root_path, file)
            for file in os.listdir(self.image_root_path)
            if file.endswith("tar")
        ]
        self.meta = []
        for tar in tarfiles:
            with tarfile.open(tar, "r") as file:
                self.meta.extend(
                    [
                        {"tar": tar, "image": member.name}
                        for member in file.getmembers()
                        if member.name.endswith(config.image_ext)
                    ]
                )

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

        self.cond_processor = CLIPImageProcessor.from_pretrained(
            config.condition_model_path
        )

        self.drop_probability: float = config.drop_probability

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Dict:
        item = self.meta[index]
        raw_image = Image.open(TarIO.TarIO(item["tar"], item["image"])).convert("RGB")

        # for vae encoder
        pixel_values = self.vae_processor(raw_image) if self.mode == "train" else None

        # for inference
        ground_truth: Union[torch.FloatTensor, None] = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val" or self.mode == "test"
            else None
        )

        # for vision condition
        conditions: torch.FloatTensor = self.cond_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values

        # handle drop to enhance classifier-free guidance
        drops = torch.rand(1) < self.drop_probability if self.mode == "train" else None

        data = {"conditions": conditions.squeeze(dim=0)}

        if self.mode == "val" or self.mode == "test":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
            data["image_indexes"] = index

        elif self.mode == "train":
            data["pixel_values"] = pixel_values
            data["drops"] = drops

        return data
