import os
import ssl
from typing import List, Callable, Dict
from os.path import expanduser
from urllib.request import urlretrieve
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import numpy as np
import torchmetrics
import open_clip
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import ViT_H_14_Weights, vit_h_14
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from omegaconf import OmegaConf, DictConfig

# ssl._create_default_https_context = ssl._create_unverified_context


GPU = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)


class ImageDataset(Dataset):
    def __init__(
        self,
        gen_image_directory: str,
        gen_image_ext: str = "png",
        processor: Callable = None,
    ) -> None:
        super().__init__()

        self.image_root = gen_image_directory

        self.meta: List = [
            file
            for file in os.listdir(gen_image_directory)
            if file.endswith(gen_image_ext)
        ]

        self.processor = processor

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index) -> torch.FloatTensor:
        image_name: str = self.meta[index]
        image_path = os.path.join(self.image_root, image_name)
        gen_image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(gen_image)

        return pixel_values


class ImageDatasetWithRef(Dataset):
    def __init__(
        self,
        gen_image_directory: str,
        ref_image_directory: str,
        gen_image_ext: str = "png",
        ref_image_ext: str = "JPEG",
        processor: Callable = None,
    ) -> None:
        super().__init__()

        self.image_root = gen_image_directory
        self.ref_image_root = ref_image_directory

        self.meta: List = [
            file
            for file in os.listdir(gen_image_directory)
            if file.endswith(gen_image_ext)
        ]
        self.ref_image_ext = ref_image_ext
        self.processor = processor

    def __len__(self):
        return len(self.meta)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if isinstance(self.processor, CLIPImageProcessor):
            pixel_values = self.processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(dim=0)
        else:
            pixel_values = self.processor(image)

        return pixel_values

    def __getitem__(self, index) -> torch.FloatTensor:
        image_name: str = self.meta[index]
        image_path = os.path.join(self.image_root, image_name)
        gen_image = Image.open(image_path).convert("RGB")
        pixel_values = self._process_image(gen_image)

        # Format:  {imageNet id} - {subject id} - {sample id} - {.png}
        ref_name = image_name.split("-")[0]
        ref_path = os.path.join(
            self.ref_image_root, ".".join([ref_name, self.ref_image_ext])
        )
        ref_image = Image.open(ref_path).convert("RGB")
        ref_values = self._process_image(ref_image)

        return pixel_values, ref_values


def get_aesthetic_model(clip_model="vit_l_14"):
    """
    Copied from https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb
    """
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


@torch.inference_mode()
def n_way_top_k_acc(
    pred: torch.Tensor, class_id: int, n_way: int, num_trials=40, top_k=1
):
    """
    Copied from Implementation of DreamDiffusion:
    https://github.com/bbaaii/DreamDiffusion/blob/main/code/eval_metrics.py
    """
    pick_range = [i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for _ in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = torchmetrics.functional.accuracy(
            pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device), top_k=top_k
        )
        acc_list.append(acc.cpu())
    return np.mean(acc_list)


def get_aesthetic_score(
    image_directory: str, image_ext: str = "png", batch_size: int = 16
):
    amodel = get_aesthetic_model(clip_model="vit_l_14")
    amodel.eval()

    model, _, processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval()

    amodel.to(GPU)
    model.to(GPU)

    dataset = ImageDataset(image_directory, image_ext, processor)
    dataloader = DataLoader(dataset, batch_size)

    result: List[torch.Tensor] = []
    for pixel_values in tqdm(dataloader, desc="Calculating [aesthetic_score]..."):
        pixel_values = pixel_values.to(GPU)

        with torch.inference_mode():
            image_features = model.encode_image(pixel_values)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = amodel(image_features)
            result.append(prediction)

    mean_result = torch.cat(result, dim=0).mean(dim=0).cpu().item()

    del model, amodel

    return mean_result


def get_nway_topk_accuracy(
    ref_image_directory: str,
    gen_image_directory: str,
    image_ext: str = "png",
    ref_image_ext: str = "JPEG",
    n_way: int = 50,
    top_k: int = 1,
    num_trials: int = 40,
):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    processor = weights.transforms()

    model = model.to(GPU)
    model = model.eval()

    dataset = ImageDatasetWithRef(
        gen_image_directory, ref_image_directory, image_ext, ref_image_ext, processor
    )
    dataloader = DataLoader(dataset, 1)
    result = []

    for pixel_values, ref_values in tqdm(
        dataloader, desc=f"Calculating [{n_way}_way_top_{top_k}_accuracy]..."
    ):
        pixel_values: torch.Tensor = pixel_values.to(GPU)
        ref_values: torch.Tensor = ref_values.to(GPU)

        with torch.inference_mode():
            ref_class_id = model(ref_values).squeeze(0).softmax(0).argmax().item()
            pred_out = model(pixel_values).squeeze(0).softmax(0).detach()

        result.append(
            torch.tensor(
                n_way_top_k_acc(
                    pred_out,
                    ref_class_id,
                    n_way=n_way,
                    num_trials=num_trials,
                    top_k=top_k,
                )
            )
        )

    mean_result = torch.stack(result, dim=0).mean(dim=0).cpu().item()

    del model

    return mean_result


def get_clip_similarity(
    clip_model_path: str,
    ref_image_directory: str,
    gen_image_directory: str,
    image_ext: str = "png",
    ref_image_ext: str = "JPEG",
    batch_size: int = 16,
):
    model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
    model.eval()
    model.to(GPU)
    processor = CLIPImageProcessor.from_pretrained(clip_model_path)

    dataset = ImageDatasetWithRef(
        gen_image_directory, ref_image_directory, image_ext, ref_image_ext, processor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    result = []

    for pixel_values, ref_values in tqdm(
        dataloader, desc=f"Calculating [clip_similarity]..."
    ):
        pixel_values: torch.Tensor = pixel_values.to(GPU)
        ref_values: torch.Tensor = ref_values.to(GPU)

        with torch.inference_mode():
            ref_embeds = model(ref_values).image_embeds
            image_embeds = model(pixel_values).image_embeds

            result.append(
                nn.functional.cosine_similarity(image_embeds, ref_embeds, dim=-1)
            )

    mean_result = torch.cat(result, dim=0).mean(dim=0).cpu().item()

    del model
    return mean_result


TASK2FUNCTION: Dict[str, Callable] = {
    "aesthetic_score": get_aesthetic_score,
    "nway_topk_accuracy": get_nway_topk_accuracy,
    "clip_similarity": get_clip_similarity,
}


def main(config: DictConfig):
    result = {}

    for name in TASK2FUNCTION.keys():
        args = config.get(name, None)
        if args is not None:
            args = OmegaConf.to_object(args)

            func = TASK2FUNCTION[name]

            result[name] = func(**args)

    for name, value in result.items():
        print(f"[{name}]: {value}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate.yml", type=str)
    config = OmegaConf.load(parser.parse_args().config)
    main(config)
