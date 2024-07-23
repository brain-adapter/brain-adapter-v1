import os
from typing import Dict, Optional, Union, List
import random

import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from PIL import Image
import pandas
from transformers import CLIPImageProcessor, CLIPTokenizer
from torchvision import transforms

from model.evaluation import resize_images


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
        self, eeg_values: torch.Tensor, return_batches: Optional[bool] = None, **kwargs
    ) -> torch.Tensor:
        return_batches = (
            return_batches if return_batches is not None else True
        )  # default to return in batch form

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

        self.clip_embeds = (
            torch.load(config.clip_embeds_path)
            if config.get("clip_embeds_path") is not None
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

        if self.clip_embeds is not None:
            data["clip_embeds"] = self.clip_embeds[item["image"]]

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
        self.text_processor = CLIPTokenizer.from_pretrained(
            config.diffusion_model_path, subfolder="tokenizer"
        )

        self.image_root_path = config.image_root_path
        self.resolution = config.get("resolution", 512)

        # handle drop
        self.eeg_drop_prob = config.get("eeg_drop_prob", 0.1)

        if mode == "val":
            self.splitter = random.sample(
                self.splitter, config.get("num_validation_images", 4)
            )
            self.dataset = [
                self.dataset[i] for i in range(len(self.dataset)) if i in self.splitter
            ]

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
            if self.mode == "val"
            else None
        )

        drop = torch.rand(1) < self.eeg_drop_prob

        # no text will be used
        text = ""
        input_ids: torch.Tensor = self.text_processor(
            text,
            max_length=self.text_processor.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        data = {
            "pixel_values": pixel_values,
            "condition_inputs": self.eeg_processor(item["eeg"], return_batches=False),
            "input_ids": input_ids.squeeze(dim=0),
            "drops": drop,
        }

        if self.mode == "val":
            data["ground_truth"] = ground_truth
        elif self.mode == "test":
            data["image_indexes"] = self.images[item["image"]]

        return data


class ImageTextDataset(Dataset):
    def __init__(self, mode: str, config: DictConfig):
        super().__init__()
        self.config = config
        self.image_root_path = config.image_root_path[mode]
        self.resolution = config.get("resolution", 512)
        self.mode = mode

        self.meta: pandas.DataFrame = pandas.read_parquet(
            config.meta_files[mode], engine="pyarrow"
        )
        self.diffusion_processor = transforms.Compose(
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
        self.clip_processor = CLIPImageProcessor.from_pretrained(config.clip_model_path)
        self.text_processor: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            config.diffusion_model_path, subfolder="tokenizer"
        )

        # handle drop to enhance classifier-free guidance
        self.text_drop_prob = config.get("text_drop_prob", 1.0)
        self.image_drop_prob = config.get("image_drop_prob", 0.1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> Dict:
        image_path = os.path.join(
            self.image_root_path, self.meta.iloc[index]["local_file"]
        )
        raw_image = Image.open(image_path).convert("RGB")

        clip_pixel_values: torch.Tensor = self.clip_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values

        diffusion_pixel_values = self.diffusion_processor(raw_image)
        ground_truth: Union[torch.Tensor, None] = (
            resize_images(raw_image, new_size=self.resolution, convert_to_tensor=True)
            if self.mode == "val"
            else None
        )

        drop_mask = torch.rand(2) < torch.tensor(
            [self.image_drop_prob, self.text_drop_prob]
        )

        drop = drop_mask[0]
        text = "" if drop_mask[1] else self.meta.iloc[index]["caption"]

        input_ids: torch.Tensor = self.text_processor(
            text,
            max_length=self.text_processor.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        data = {
            "pixel_values": diffusion_pixel_values,
            "condition_inputs": clip_pixel_values.squeeze(dim=0),
            "input_ids": input_ids.squeeze(dim=0),
            "drops": drop,
        }

        if self.mode == "val":
            data["ground_truth"] = ground_truth.squeeze(dim=0)
        elif self.mode == "test":
            data["image_indexes"] = index

        return data


class ImageNetDataset(Dataset):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.image_root: str = config.image_root_path
        self.image_meta: List[str] = config.meta_data
        self.image_ext: str = config.image_ext
        self.image_processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(
            config.clip_model_path
        )

    def __len__(self):
        return len(self.image_meta)

    def __getitem__(self, index) -> Dict:
        image_name = self.image_meta[index]
        folder_name = image_name.split("_")[0]
        image_path = os.path.join(
            self.image_root, folder_name, ".".join([image_name, self.image_ext])
        )

        raw_image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values.squeeze(dim=0)

        return {"pixel_values": pixel_values}
