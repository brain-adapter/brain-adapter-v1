import os
from typing import List, Dict
from argparse import ArgumentParser, Namespace
from pathlib2 import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers.models.clip import CLIPVisionModelWithProjection, CLIPImageProcessor


class ImageDataset(Dataset):
    def __init__(self, clip_model_path: str, image_root: str, image_ext: str) -> None:
        super().__init__()

        self.image_root = image_root
        self.meta: List = [
            file for file in os.listdir(image_root) if file.endswith(image_ext)
        ]

        self.processor = CLIPImageProcessor.from_pretrained(clip_model_path)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        image_name: str = self.meta[index]
        image_path = os.path.join(self.image_root, image_name)
        raw_image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            raw_image, return_tensors="pt"
        ).pixel_values.squeeze(dim=0)

        image_id = Path(image_name).stem

        return image_id, pixel_values


def main(config: Namespace):
    model = CLIPVisionModelWithProjection.from_pretrained(config.clip_model_path)
    dataset = ImageDataset(config.clip_model_path, config.image_root, config.image_ext)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    gpu = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    result: Dict[str, Dict] = {}

    # get image embeds
    for samples in tqdm(dataloader):
        image_ids, pixel_values = samples
        pixel_values = pixel_values.to(gpu)

        with torch.no_grad():
            image_embeds = model(pixel_values).image_embeds
            image_embeds = image_embeds.cpu()

        for image_id, embeds in zip(image_ids, image_embeds):
            result[image_id] = {
                "image_embeds": embeds,
                "label": image_id.split("_")[0],
                "image": image_id,
                "dist": {},  # for accelerating
                "sim": {},
            }

    # get distance map for every image embeds
    for img_id, value in tqdm(result.items()):
        label = value["label"]
        anchor = value["image_embeds"]

        dist_map: Dict = value["dist"]
        sim_map: Dict = value["sim"]

        others = [result[key] for key in result.keys() if not key.startswith(label)]

        for sample in others:
            sample_id = sample["image"]
            if (
                dist_map.get(sample_id, None) is None
                and sim_map.get(sample_id, None) is None
            ):
                target_embeds = sample["image_embeds"]
                dist = nn.functional.pairwise_distance(anchor, target_embeds)

                dist_map[sample_id] = dist
                sample["dist"][img_id] = dist

                sim = nn.functional.cosine_similarity(anchor, target_embeds, dim=-1)
                sim_map[sample_id] = sim
                sample["sim"][img_id] = sim

    for item in result.values():
        # filter top-5
        dist_map = item["dist"]

        dist_list = [dist for _, dist in dist_map.items()]
        dist_ids = [index for index, _ in dist_map.items()]
        # top5 lowest
        values, indices = torch.topk(torch.stack(dist_list), k=5, largest=False)
        item["dist"] = {dist_ids[index]: value for index, value in zip(indices, values)}
        item["top1_dist"] = dist_ids[indices[0]]

        sim_map = item["sim"]

        sim_list = [sim for _, sim in sim_map.items()]
        sim_ids = [index for index, _ in sim_map.items()]
        # top5 highest
        values, indices = torch.topk(torch.stack(sim_list), k=5)
        item["sim"] = {sim_ids[index]: value for index, value in zip(indices, values)}
        item["top1_sim"] = sim_ids[indices[0]]

    torch.save(result, "image_embeds.pth")

    # visualize
    for item in result.values():
        # filter top-5
        item.pop("image_embeds")
        item["dist"] = {key: item["dist"][key].item() for key in item["dist"].keys()}
        item["sim"] = {key: item["sim"][key].item() for key in item["sim"].keys()}

    import json

    with open("result.json", "w") as f:
        json.dump(result, f)

    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--image-root",
        default="/root/autodl-tmp/data/eeg-imagenet/images",
        type=str,
    )
    parser.add_argument(
        "--clip-model-path",
        default="/root/autodl-tmp/pretrained/clip-vit-large-patch14-336",
        type=str,
    )
    parser.add_argument(
        "--image-ext",
        default="JPEG",
        type=str,
    )

    args = parser.parse_args()

    main(args)
