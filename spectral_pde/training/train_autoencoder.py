"""Train autoencoder models on PDE data.

This script trains autoencoders in both spatial and spectral representations and saves
checkpoints for later use.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from spectral_pde.config import Config, get_config
from spectral_pde.models.autoencoder import build_autoencoder
from spectral_pde.utils.dataset import PDEDataset


def train_autoencoder(
    trajectories: np.ndarray,
    spatial_shape: Tuple[int, ...],
    config: Config,
    mode: str = "spatial",
    dim: int = 1,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Train an autoencoder on the provided trajectory data."""

    dataset = PDEDataset(trajectories, mode=mode, dim=dim)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = build_autoencoder(
        dim=dim,
        spatial_shape=spatial_shape,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        spectral=(mode == "spectral"),
    )
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, config.num_epochs_ae + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # ensure channel-first for 1D/2D conv autoencoders
            if dim == 1 and x.ndim == 2:
                # spatial 1D: (batch, N) -> (batch, 1, N)
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
            elif dim == 2:
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1)
                else:
                    x = x.permute(0, 3, 1, 2)
                    y = y.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(dataset)
        print(f"[AE {mode}] Epoch {epoch}/{config.num_epochs_ae} - loss {avg_loss:.6f}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"autoencoder_{dim}d_{mode}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved autoencoder checkpoint: {checkpoint_path}")

    return model


if __name__ == "__main__":
    import numpy as np

    cfg = get_config()
    # for a quick demo, simulate a small 1D heat equation dataset
    from spectral_pde.data.pde_simulation import generate_trajectories

    trajectories = generate_trajectories(
        pde="heat1d",
        num_trajectories=16,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_1d,
        dt=cfg.dt,
        dx=cfg.dx,
        alpha=cfg.diffusion_coeff,
        seed=cfg.seed,
    )

    train_autoencoder(
        trajectories, (cfg.grid_size_1d,), cfg, mode="spatial", dim=1
    )
    train_autoencoder(
        trajectories, (cfg.grid_size_1d,), cfg, mode="spectral", dim=1
    )
