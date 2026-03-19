"""Simplified Fourier Neural Operator (FNO) layer implementations.

The idea is to operate in the Fourier domain on a small set of low-frequency modes
with learnable complex weights, then combine with a pointwise linear mapping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer that operates on low-frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # weight: (in, out, modes, 2) for complex weights (real/imag)
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, 2) * 0.1)

    def compl_mul1d(self, input, weights):
        # (batch, in, modes) complex * (in, out, modes) complex -> (batch, out, modes)
        # input: complex tensor of shape (batch, in, modes)
        # weights: complex tensor of shape (in, out, modes)
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n)
        batchsize, channels, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        # take first modes
        x_ft_low = x_ft[:, :, : self.modes]
        weight_complex = torch.complex(self.weights[..., 0], self.weights[..., 1])
        out_ft = self.compl_mul1d(x_ft_low, weight_complex)
        # pad the remaining modes with zeros
        out_ft_full = torch.zeros((batchsize, self.out_channels, x_ft.shape[-1]), dtype=torch.complex64, device=x.device)
        out_ft_full[:, :, : self.modes] = out_ft
        x = torch.fft.irfft(out_ft_full, n=n, dim=-1)
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer that operates on low-frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes, 2) * 0.1)

    def compl_mul2d(self, input, weights):
        # input: (batch, in, modes1, modes2)
        # weights: (in, out, modes1, modes2)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, h, w)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            (batchsize, self.out_channels, x_ft.shape[-2], x_ft.shape[-1]),
            dtype=torch.complex64,
            device=x.device,
        )
        # use low frequency modes only
        m = self.modes
        x_ft_low = x_ft[:, :, :m, :m]
        weight_complex = torch.complex(self.weights[..., 0], self.weights[..., 1])
        out_ft[:, :, :m, :m] = self.compl_mul2d(x_ft_low, weight_complex)
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), dim=(-2, -1))
        return x


class FNO1D(nn.Module):
    """Simple 1D Fourier Neural Operator model."""

    def __init__(self, modes: int, width: int, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "spectral": SpectralConv1d(width, width, modes),
                "pointwise": nn.Conv1d(width, width, 1),
            }))
        self.project_in = nn.Conv1d(1, width, 1)
        self.project_out = nn.Conv1d(width, 1, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n)
        x = self.project_in(x)
        for layer in self.layers:
            x1 = layer["spectral"](x)
            x2 = layer["pointwise"](x)
            x = self.activation(x1 + x2)
        x = self.project_out(x)
        return x


class FNO2D(nn.Module):
    """Simple 2D Fourier Neural Operator model."""

    def __init__(self, modes: int, width: int, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "spectral": SpectralConv2d(width, width, modes),
                "pointwise": nn.Conv2d(width, width, 1),
            }))
        self.project_in = nn.Conv2d(1, width, 1)
        self.project_out = nn.Conv2d(width, 1, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, h, w)
        x = self.project_in(x)
        for layer in self.layers:
            x1 = layer["spectral"](x)
            x2 = layer["pointwise"](x)
            x = self.activation(x1 + x2)
        x = self.project_out(x)
        return x
