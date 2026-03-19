"""Autoencoder models for spatial and spectral PDE data."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Autoencoder1D(nn.Module):
    """Fully connected autoencoder for 1D data (spatial or spectral)."""

    def __init__(
        self,
        input_size: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        channels: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels
        in_dim = input_size * channels

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
            nn.Unflatten(1, (channels, input_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class Autoencoder2D(nn.Module):
    """Convolutional autoencoder for 2D data."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_dim: int = 32,
        hidden_dim: int = 32,
        channels: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels

        # encoder: downsample spatially
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_dim * input_size[0] * input_size[1], latent_dim),
        )

        # decoder: project back and reshape
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * input_size[0] * input_size[1]),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (hidden_dim, input_size[0], input_size[1])),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        out = self.decode(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def build_autoencoder(
    dim: int,
    spatial_shape: Tuple[int, ...],
    latent_dim: int,
    hidden_dim: int,
    spectral: bool = False,
) -> nn.Module:
    """Construct an autoencoder for 1D or 2D data.

    Args:
        dim: spatial dimension (1 or 2).
        spatial_shape: shape of the spatial grid (e.g., (N,) or (N, N)).
        latent_dim: latent dimension.
        hidden_dim: hidden dimension for encoder/decoder.
        spectral: whether the input is spectral (2 channels) or spatial (1 channel).
    """
    channels = 2 if spectral else 1
    if dim == 1:
        # spectral representation reduces size due to rfft symmetry
        input_size = spatial_shape[0] // 2 + 1 if spectral else spatial_shape[0]
        return Autoencoder1D(input_size, latent_dim=latent_dim, hidden_dim=hidden_dim, channels=channels)
    elif dim == 2:
        if spectral:
            # rfft2 produces W//2+1 frequency bins in the last dimension
            input_size = (spatial_shape[0], spatial_shape[1] // 2 + 1)
        else:
            input_size = (spatial_shape[0], spatial_shape[1])
        return Autoencoder2D(input_size, latent_dim=latent_dim, hidden_dim=hidden_dim, channels=channels)
    else:
        raise ValueError("dim must be 1 or 2")
