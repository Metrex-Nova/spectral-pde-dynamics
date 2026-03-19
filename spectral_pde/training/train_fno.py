"""Train a Fourier Neural Operator (FNO) model on PDE data."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from spectral_pde.config import Config, get_config
from spectral_pde.models.fno_layer import FNO1D, FNO2D
from spectral_pde.utils.dataset import PDEDataset


def train_fno(
    trajectories: np.ndarray,
    spatial_shape: Tuple[int, ...],
    config: Config,
    dim: int = 1,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Train an FNO model for 1D or 2D PDE dynamics."""

    device = torch.device(config.device)
    dataset = PDEDataset(trajectories, mode="spatial", dim=dim)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if dim == 1:
        model = FNO1D(modes=config.fno_modes_1d, width=config.fno_width, depth=config.fno_layers)
    else:
        model = FNO2D(modes=config.fno_modes_2d, width=config.fno_width, depth=config.fno_layers)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, config.num_epochs_fno + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # input shape for FNO: (batch, 1, n) or (batch, 1, h, w)
            if dim == 1:
                x_in = x.unsqueeze(1)
            else:
                x_in = x.unsqueeze(1)

            pred = model(x_in)
            loss = criterion(pred, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]

        avg = total / len(dataset)
        print(f"[FNO {dim}D] Epoch {epoch}/{config.num_epochs_fno} loss {avg:.6f}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"fno_{dim}d.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved FNO checkpoint: {checkpoint_path}")

    return model


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

    train_fno(trajectories, (cfg.grid_size_1d,), cfg, dim=1)
