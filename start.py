import os
from tqdm import tqdm
from pathlib2 import Path
from argparse import ArgumentParser, Namespace
import random
import shutil

import torch
import pandas
import lightning
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection

from data.dataset import ImageNetDataset


def create_embeds(
    data_root_path: str,
    clip_model_path: str,
    save_dir_path: str,
    version: str = "clip-large-patch14",
    clip_skip: int = 1,
):
    """
    Create CLIP Vision Model embeds in advance to cut off training cost
    """
    image_meta = torch.load(os.path.join(data_root_path, "eeg_5_95_std.pth"))["images"]
    image_net_dataset = ImageNetDataset(
        OmegaConf.create(
            {
                "image_root_path": os.path.join(data_root_path, "images"),
                "meta_data": image_meta,
                "clip_model_path": clip_model_path,
                "image_ext": "JPEG",
            }
        )
    )
    imagenet_dataloader = DataLoader(dataset=image_net_dataset, batch_size=16, shuffle=False)

    gpu = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
    clip_model.to(gpu)

    clip_embeds_ = []

    for batch in tqdm(imagenet_dataloader, desc=f"Creating embeds for {version}"):
        pixel_values: torch.Tensor = batch["pixel_values"].to(gpu)
        with torch.inference_mode():
            clip_embeds_batch = clip_model(
                pixel_values,  return_dict=True
            ).image_embeds

        clip_embeds_.append(clip_embeds_batch.cpu())

    clip_embeds = torch.cat(clip_embeds_, dim=0)

    # save embeds into binary files
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    torch.save(
        clip_embeds, os.path.join(save_dir_path, "-".join([version, "embeds.bin"]))
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_val_meta(
    data_root_path: str, num_images_per_class: int = 4, save_dir_path: str = ""
):
    """
    For Validation process of Training Vision Adapter
    """
    image_meta = torch.load(os.path.join(data_root_path, "eeg_5_95_std.pth"))["images"]
    label_meta = torch.load(os.path.join(data_root_path, "eeg_5_95_std.pth"))["labels"]

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    image_directory = os.path.join(save_dir_path, "images")
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    metadata_val = []
    for label in tqdm(label_meta, desc="Creating validation data..."):
        image_columns = [image for image in image_meta if image.startswith(label)]
        image_columns = random.sample(image_columns, k=num_images_per_class)

        for image in image_columns:
            image_file = ".".join([image, "JPEG"])
            image_src = os.path.join(data_root_path, "images", label, image_file)
            image_dst = os.path.join(save_dir_path, "images", image_file)
            shutil.copy(image_src, image_dst)

            metadata_val.append(
                {
                    "local_file": image_file,
                    "caption": "",  # do not need caption
                }
            )

    # save validaion meta file
    pandas.DataFrame(metadata_val).to_parquet(
        os.path.join(save_dir_path, "meta-val.parquet"), engine="pyarrow"
    )


def main(args: Namespace):
    lightning.seed_everything(args.seed)

    for clip_model_path in args.clip_model_list:
        create_embeds(
            data_root_path=args.data_root_path,
            clip_model_path=clip_model_path,
            save_dir_path=args.embeds_file_path,
            clip_skip=args.clip_skip,
            version=Path(clip_model_path).stem,
        )

    # create_val_meta(
    #     data_root_path=args.data_root_path, save_dir_path=args.validation_directory
    # )


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare everything for the project")
    parser.add_argument(
        "--data-root-path", type=str, default="/root/autodl-tmp/data/eeg-imagenet"
    )
    parser.add_argument("--seed", type=int, default="2024")
    parser.add_argument(
        "--clip-model-list",
        nargs="+",
        default=["/root/autodl-tmp/pretrained/clip-vit-l-14"],
    )
    parser.add_argument(
        "--embeds-file-path",
        type=str,
        default="/root/autodl-tmp/data/eeg-imagenet/embeds",
    )
    parser.add_argument(
        "--validation-directory", type=str, default="/root/autodl-tmp/data/ada10m-en/validation"
    )
    parser.add_argument("--clip-skip", type=int, default=1)

    args = parser.parse_args()
    main(args)
