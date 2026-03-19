"""Train latent dynamics models on encoded PDE trajectories."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from spectral_pde.config import Config, get_config
from spectral_pde.models.latent_dynamics import LatentDynamics
from spectral_pde.models.autoencoder import build_autoencoder
from spectral_pde.utils.dataset import PDEDataset


def train_latent_dynamics(
    trajectories: np.ndarray,
    spatial_shape: Tuple[int, ...],
    config: Config,
    mode: str = "spatial",
    dim: int = 1,
    use_lstm: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Train latent dynamics model using an autoencoder.

    Returns (autoencoder, dynamics_model, optimizer)
    """
    device = torch.device(config.device)

    # Build and train autoencoder first
    ae = build_autoencoder(
        dim=dim,
        spatial_shape=spatial_shape,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        spectral=(mode == "spectral"),
    ).to(device)

    # Train autoencoder briefly (warm start) if not already trained.
    # For simplicity we train here from scratch (can use checkpoints externally).
    dataset_ae = PDEDataset(trajectories, mode=mode, dim=dim)
    loader_ae = DataLoader(dataset_ae, batch_size=config.batch_size, shuffle=True)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, config.num_epochs_ae + 1):
        ae.train()
        total = 0.0
        for x, y in loader_ae:
            x = x.to(device)
            y = y.to(device)
            if dim == 1 and x.ndim == 2:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
            opt_ae.zero_grad()
            out = ae(x)
            loss = criterion(out, y)
            loss.backward()
            opt_ae.step()
            total += loss.item() * x.shape[0]
        avg = total / len(dataset_ae)
        print(f"[AE warm] Epoch {epoch}/{config.num_epochs_ae} loss {avg:.6f}")

    # Freeze autoencoder for dynamics training
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Dynamics model
    dynamics = LatentDynamics(config.latent_dim, hidden_dim=config.hidden_dim, use_lstm=use_lstm).to(device)
    opt_dyn = torch.optim.Adam(dynamics.parameters(), lr=config.learning_rate)

    # Data loader for dynamics (pairs)
    dataset_dyn = PDEDataset(trajectories, mode=mode, dim=dim)
    loader_dyn = DataLoader(dataset_dyn, batch_size=config.batch_size, shuffle=True)

    for epoch in range(1, config.num_epochs_dynamics + 1):
        dynamics.train()
        total = 0.0
        for x, y in loader_dyn:
            x = x.to(device)
            y = y.to(device)
            if dim == 1 and x.ndim == 2:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
            elif dim == 2:
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                    y = y.unsqueeze(1)
                else:
                    x = x.permute(0, 3, 1, 2)
                    y = y.permute(0, 3, 1, 2)

            z = ae.encode(x)
            z_pred, _ = dynamics(z)
            y_pred = ae.decode(z_pred)

            loss = criterion(y_pred, y)
            opt_dyn.zero_grad()
            loss.backward()
            opt_dyn.step()
            total += loss.item() * x.shape[0]

        avg = total / len(dataset_dyn)
        print(f"[Dynamics {mode}] Epoch {epoch}/{config.num_epochs_dynamics} loss {avg:.6f}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"latent_dynamics_{dim}d_{mode}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "ae": ae.state_dict(),
        "dyn": dynamics.state_dict(),
    }, checkpoint_path)
    print(f"Saved latent dynamics checkpoint: {checkpoint_path}")

    return ae, dynamics, opt_dyn


if __name__ == "__main__":
    from spectral_pde.data.pde_simulation import generate_trajectories

    cfg = get_config()
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

    train_latent_dynamics(
        trajectories,
        (cfg.grid_size_1d,),
        cfg,
        mode="spatial",
        dim=1,
        use_lstm=False,
    )
