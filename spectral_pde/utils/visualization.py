"""Visualization utilities for comparing PDE trajectories and model predictions."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_1d_evolution(
    true: np.ndarray,
    pred: np.ndarray,
    timesteps: Sequence[int] = (0, -1),
    title: str = "1D evolution",
    save_path: Optional[str] = None,
) -> None:
    """Plot 1D solutions at given timesteps."""
    plt.figure(figsize=(10, 4))
    n = true.shape[-1]
    x = np.arange(n)

    for t in timesteps:
        plt.plot(x, true[t], "-", label=f"true t={t}")
        plt.plot(x, pred[t], "--", label=f"pred t={t}")

    plt.title(title)
    plt.xlabel("grid index")
    plt.ylabel("u")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_error_over_time(
    true: np.ndarray,
    pred: np.ndarray,
    title: str = "Error over time",
    save_path: Optional[str] = None,
) -> None:
    """Plot MSE error at each time step."""
    errors = np.mean((true - pred) ** 2, axis=tuple(range(1, true.ndim)))
    plt.figure(figsize=(6, 4))
    plt.plot(errors, "-o")
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_2d_heatmaps(
    true: np.ndarray,
    pred: np.ndarray,
    timesteps: Sequence[int] = (0, -1),
    title: str = "2D heatmaps",
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D heatmaps (true vs predicted) for selected timesteps."""
    n = len(timesteps)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, t in enumerate(timesteps):
        vmin = min(true[t].min(), pred[t].min())
        vmax = max(true[t].max(), pred[t].max())
        im0 = axes[i, 0].imshow(true[t], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i, 0].set_title(f"true t={t}")
        fig.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].imshow(pred[t], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i, 1].set_title(f"pred t={t}")
        fig.colorbar(im1, ax=axes[i, 1])

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_spectral_vs_spatial(
    spatial: np.ndarray,
    spectral: np.ndarray,
    title: str = "Spectral vs Spatial",
    save_path: Optional[str] = None,
) -> None:
    """Plot a comparison between spatial and spectral representations.

    The spectral data should be of shape (..., 2) with real/imag channels.
    """
    # show magnitude spectrum
    mag = np.sqrt(spectral[..., 0] ** 2 + spectral[..., 1] ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(spatial.flatten(), label="spatial")
    axes[0].set_title("Spatial signal (flattened)")
    axes[0].legend()

    axes[1].plot(mag.flatten(), label="spectral magnitude")
    axes[1].set_title("Spectral magnitude (flattened)")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
