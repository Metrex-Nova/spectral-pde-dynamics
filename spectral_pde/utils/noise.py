"""Noise utilities for robustness experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
except ImportError:  # type: ignore
    torch = None  # type: ignore


def add_gaussian_noise(x, std: float = 0.01, clip: bool = False):
    """Add Gaussian noise to a tensor or numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        noise = torch.randn_like(x) * std
        out = x + noise
        if clip:
            return out.clamp(-1.0, 1.0)
        return out

    x = np.asarray(x)
    noise = np.random.randn(*x.shape) * std
    out = x + noise
    if clip:
        return np.clip(out, -1.0, 1.0)
    return out


def gaussian_noise_fn(std: float = 0.01, clip: bool = False) -> Callable:
    """Return a function that adds Gaussian noise with specified standard deviation."""

    def fn(x):
        return add_gaussian_noise(x, std=std, clip=clip)

    return fn
