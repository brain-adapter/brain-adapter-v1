import os
from argparse import ArgumentParser, Namespace
from typing import Type
from pathlib2 import Path

import torch
from omegaconf import OmegaConf

from model.activations import get_class
from model.lightnings import LitBaseModel


def resolve_model(checkpoint_path: str) -> LitBaseModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])

    lightning_class: Type[LitBaseModel] = get_class(config.lightning.name)
    model = lightning_class.load_from_checkpoint(checkpoint_path)

    return model

def main(args:Namespace):
    src: str = args.lit_directory
    dst: str = args.save_directory
    del_lit: bool = args.del_lit

    lit_files = [file for file in os.listdir(src) if file.endswith("ckpt")]

    for file in lit_files:
        ckpt_path = os.path.join(src, file)
        model = resolve_model(ckpt_path)

        model.save_pretrained(os.path.join(dst, Path(file).stem))

        if del_lit:
            os.remove(ckpt_path)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--lit-directory", default="ckpts", type=str)
    parser.add_argument("--save-directory", default="pretrained", type=str)
    parser.add_argument("--del-lit", default=False, type=bool)

    args = parser.parse_args()

    main(args)
