"""Experiment: compare spatial vs spectral autoencoder representations."""

from __future__ import annotations

import os

import numpy as np
import torch

from spectral_pde.config import get_config
from spectral_pde.data.pde_simulation import generate_trajectories
from spectral_pde.models.autoencoder import build_autoencoder
from spectral_pde.utils.dataset import PDEDataset
from spectral_pde.utils.visualization import plot_error_over_time, plot_spectral_vs_spatial


def run_experiment():
    cfg = get_config()
    # generate a small test trajectory
    trajectories = generate_trajectories(
        pde="heat1d",
        num_trajectories=4,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_1d,
        dt=cfg.dt,
        dx=cfg.dx,
        alpha=cfg.diffusion_coeff,
        seed=cfg.seed,
    )

    # Train spatial autoencoder
    spatial_ds = PDEDataset(trajectories, mode="spatial", dim=1)
    spatial_ae = build_autoencoder(
        dim=1,
        spatial_shape=(cfg.grid_size_1d,),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        spectral=False,
    )
    # Train quickly for demo
    _train_model(spatial_ae, spatial_ds, cfg, "spatial")

    # Train spectral autoencoder
    spectral_ds = PDEDataset(trajectories, mode="spectral", dim=1)
    spectral_ae = build_autoencoder(
        dim=1,
        spatial_shape=(cfg.grid_size_1d,),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        spectral=True,
    )
    _train_model(spectral_ae, spectral_ds, cfg, "spectral")

    # Compare reconstruction error for a single trajectory
    x, y = spatial_ds[0]
    with torch.no_grad():
        spatial_pred = spatial_ae(x.unsqueeze(0)).squeeze(0).numpy()
        x_spec, _ = spectral_ds[0]
        spectral_pred = spectral_ae(x_spec.unsqueeze(0)).squeeze(0).numpy()

    # Plot spectral vs spatial representations for the first time step
    plot_spectral_vs_spatial(
        spatial=x.numpy(),
        spectral=x_spec.numpy(),
        title="Spatial vs Spectral (first step)",
        save_path=os.path.join("plots", "spatial_vs_spectral.png"),
    )

    # Compute and plot error over time for the first trajectory
    true_traj = trajectories[0]
    pred_traj = np.zeros_like(true_traj)
    for t in range(true_traj.shape[0] - 1):
        inp = torch.from_numpy(true_traj[t]).float().unsqueeze(0)
        pred_traj[t + 1] = spatial_ae(inp).squeeze(0).detach().cpu().numpy()

    plot_error_over_time(
        true=true_traj,
        pred=pred_traj,
        title="Spatial AE rollout error",
        save_path=os.path.join("plots", "spatial_ae_error.png"),
    )


def _train_model(model, dataset, cfg, label: str):
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    device = torch.device(cfg.device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1, cfg.num_epochs_ae + 1):
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # ensure consistent channel dimensions for 1D spatial data
            if x.ndim == 2:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)

            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        avg = total / len(dataset)
        print(f"[{label}] Epoch {epoch}/{cfg.num_epochs_ae} loss {avg:.6f}")


if __name__ == "__main__":
    run_experiment()
