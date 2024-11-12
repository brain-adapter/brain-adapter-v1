import os
from typing import List
from argparse import ArgumentParser, Namespace
from pathlib2 import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


EXTENSIONS = ["jpeg", "jpg", "webp", "png"]


def main(config: Namespace):
    new_size: int = config.new_size
    src: str = config.image_root
    dst: str = config.save_directory
    save_ext: str = config.save_ext

    image_files: List[str] = [
        file
        for file in os.listdir(src)
        if Path(file).suffix.lower().strip(".") in EXTENSIONS
    ]
    processor = transforms.Compose(
        [
            transforms.Resize(
                new_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(new_size),
        ]
    )

    if not os.path.exists(dst):
        os.makedirs(dst)

    for file in tqdm(image_files):
        path = os.path.join(src, file)
        image = Image.open(path).convert("RGB")

        resized = processor(image)

        save_path = os.path.join(dst, ".".join([Path(file).stem, save_ext]))
        resized.save(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image-root",
        default="data/eeg-imagenet/images",
        type=str,
    )

    parser.add_argument(
        "--save-directory",
        default="gen-images",
        type=str,
    )

    parser.add_argument(
        "--new-size",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--save-ext",
        default="png",
        type=str,
    )

    args = parser.parse_args()

    main(args)
