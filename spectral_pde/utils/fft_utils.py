"""FFT utilities for 1D/2D spectral representations.

Supports both numpy and torch tensors, and provides helpers to convert complex-valued
spectral representations into real-channel tensors suitable for neural networks.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # type: ignore
    torch = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _is_torch(x: ArrayLike) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def fft1d(x: ArrayLike) -> ArrayLike:
    """Compute 1D FFT (real-to-complex) along the last axis."""
    if _is_torch(x):
        return torch.fft.rfft(x, dim=-1)
    return np.fft.rfft(x, axis=-1)


def ifft1d(x: ArrayLike, n: int = None) -> ArrayLike:
    """Compute 1D inverse FFT (complex-to-real) along the last axis."""
    if _is_torch(x):
        return torch.fft.irfft(x, n=n, dim=-1)
    return np.fft.irfft(x, n=n, axis=-1)


def fft2d(x: ArrayLike) -> ArrayLike:
    """Compute 2D FFT (real-to-complex) on the last two axes."""
    if _is_torch(x):
        return torch.fft.rfft2(x, dim=(-2, -1))
    return np.fft.rfft2(x, axes=(-2, -1))


def ifft2d(x: ArrayLike, s: Tuple[int, int] = None) -> ArrayLike:
    """Compute 2D inverse FFT (complex-to-real) on the last two axes."""
    if _is_torch(x):
        return torch.fft.irfft2(x, s=s, dim=(-2, -1))
    return np.fft.irfft2(x, s=s, axes=(-2, -1))


def complex_to_chan(x: ArrayLike) -> ArrayLike:
    """Convert complex-valued FFT output into real-valued channels (real, imag)."""
    if _is_torch(x):
        return torch.stack((x.real, x.imag), dim=-1)
    return np.stack((x.real, x.imag), axis=-1)


def chan_to_complex(x: ArrayLike) -> ArrayLike:
    """Convert real-valued channel representation back into complex tensor."""
    if _is_torch(x):
        return torch.complex(x[..., 0], x[..., 1])
    return x[..., 0] + 1j * x[..., 1]


def to_spectral(x: ArrayLike, dim: int = 1) -> ArrayLike:
    """Compute the spectral representation (real+imag channels) for 1D/2D real input.

    Args:
        x: input tensor. For 1D, shape is (..., n). For 2D, shape is (..., h, w).
        dim: spatial dimension (1 or 2).
    """
    if dim == 1:
        if _is_torch(x):
            spec = fft1d(x)
        else:
            spec = fft1d(x)
        return complex_to_chan(spec)

    # 2D
    if _is_torch(x):
        spec = fft2d(x)
    else:
        spec = fft2d(x)
    return complex_to_chan(spec)


def from_spectral(x: ArrayLike, original_shape: Tuple[int, ...]) -> ArrayLike:
    """Convert spectral representation back to physical space.

    Args:
        x: real-valued spectral channels (..., 2).
        original_shape: desired output shape (excluding batch dimension).
    """
    z = chan_to_complex(x)
    if len(original_shape) == 1 or (len(original_shape) == 2 and original_shape[-2] == 1):
        # 1D
        out = ifft1d(z, n=original_shape[-1])
        if _is_torch(out) and out.ndim == 1:
            return out
        return out
    out = ifft2d(z, s=tuple(original_shape[-2:]))
    return out
