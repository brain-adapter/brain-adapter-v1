import torch
from torch import nn
import math


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
            )
        )


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1
                + torch.tanh(
                    self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))
                )
            )
        )


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "quick_gelu": QuickGELUActivation(),
    "fast_gelu": FastGELUActivation(),
    "accurate_gelu": AccurateGELUActivation(),
}

ACTIVATION_MODULES = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "quick_gelu": QuickGELUActivation,
    "fast_gelu": FastGELUActivation,
    "accurate_gelu": AccurateGELUActivation,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def get_activation_module(act_fn: str) -> nn.Module:
    """Helper function to get activation module class from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        type: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_MODULES:
        return ACTIVATION_MODULES[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
