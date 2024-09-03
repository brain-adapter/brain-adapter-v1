from typing import Union, List, Callable

import torch
import PIL
import numpy as np
import torchmetrics
from torch import nn
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


@torch.inference_mode()
def n_way_top_k_acc(pred:torch.Tensor, class_id:int, n_way:int, num_trials=40, top_k=1):
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


@torch.inference_mode()
def clip_similarity(clip_embeds_gen: torch.Tensor, clip_embeds_gt: torch.Tensor):
    return nn.functional.cosine_similarity(clip_embeds_gen, clip_embeds_gt, dim=-1).mean()


EVALUATION_FUNCTIONS = {
    "n_way_top_k_acc": n_way_top_k_acc,
    "clip_similarity": clip_similarity,
}


def get_evaluation(eva_name: str) -> Callable:
    """Helper function to get evaluation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    eva_name = eva_name.lower()
    if eva_name in EVALUATION_FUNCTIONS:
        return EVALUATION_FUNCTIONS[eva_name]
    else:
        raise ValueError(f"Unsupported evaluation function: {eva_name}")
