from typing import Union, List

import torch
import PIL
import numpy as np
from torchvision import transforms


def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            PIL.Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images

def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def resize_images(
    images: Union[PIL.Image.Image], new_size: int, convert_to_tensor: bool = False
) -> Union[List[PIL.Image.Image], torch.Tensor]:
    process_arr = [
        transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(new_size),
        transforms.ToTensor() if convert_to_tensor else None,
    ]
    processor = transforms.Compose([func for func in process_arr if func is not None])

    if isinstance(images, PIL.Image.Image):
        source = [images]
    elif isinstance(images, List):
        source = images
    else:
        raise ValueError(
            f"images can either be PIL.Image or List of PIL.Image, but got {type(images)} instead."
        )

    result = [processor(image) for image in source]
    if convert_to_tensor:
        result = torch.stack(result, dim=0)
    return result
