import os
from os.path import expanduser
from urllib.request import urlretrieve
from argparse import ArgumentParser, Namespace

import torch
import open_clip
from torch import nn
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


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


GPU = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)


def get_aesthetic_score(image: Image.Image):
    amodel = get_aesthetic_model(clip_model="vit_l_14")
    amodel.eval()

    model, _, processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval()

    amodel.to(GPU)
    model.to(GPU)
    pixel_values = processor(image).to(GPU).unsqueeze(dim=0)

    with torch.inference_mode():
        image_features = model.encode_image(pixel_values)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction:torch.Tensor = amodel(image_features).cpu().squeeze(dim=0)

    del model, amodel

    return prediction.item()


def get_clip_similarity(
    clip_model_path: str, image: Image.Image, ref_image: Image.Image
):
    model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
    processor = CLIPImageProcessor.from_pretrained(clip_model_path)

    model.eval()
    model.to(GPU)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    ref_values = processor(images=ref_image, return_tensors="pt").pixel_values

    with torch.inference_mode():
        pixel_values = pixel_values.to(GPU)
        ref_values = ref_values.to(GPU)

        ref_embeds = model(ref_values).image_embeds
        image_embeds = model(pixel_values).image_embeds

        result = (
            nn.functional.cosine_similarity(image_embeds, ref_embeds, dim=-1)
            .cpu()
            .squeeze(dim=0)
        )
    
    del model

    return result.item()


def main(args: Namespace):
    gen_image = Image.open(args.gen_image_path).convert("RGB")
    ref_image = Image.open(args.ref_image_path).convert("RGB")
    clip_model_path: str = args.clip_model_path

    aesthetic_score = get_aesthetic_score(gen_image)
    clip_similarity = get_clip_similarity(clip_model_path, gen_image, ref_image)

    print(f"aesthetic_score: [{aesthetic_score}]")
    print(f"clip_similarity: [{clip_similarity}]")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--clip-model-path",
        default="pretrained/clip-vit-base-patch32",
        type=str,
    )

    parser.add_argument(
        "--gen-image-path",
        default="gen.JPEG",
        type=str,
    )
    parser.add_argument(
        "--ref-image-path",
        default="ref.JPEG",
        type=str,
    )

    args = parser.parse_args()

    main(args)
