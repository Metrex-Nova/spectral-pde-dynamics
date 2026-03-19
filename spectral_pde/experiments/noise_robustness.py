"""Experiment: evaluate noise robustness of autoencoder models."""

from __future__ import annotations

import os

import numpy as np
import torch

from spectral_pde.config import get_config
from spectral_pde.data.pde_simulation import generate_trajectories
from spectral_pde.models.autoencoder import build_autoencoder
from spectral_pde.utils.dataset import PDEDataset
from spectral_pde.utils.noise import gaussian_noise_fn
from spectral_pde.utils.visualization import plot_error_over_time


def run_experiment():
    cfg = get_config()

    trajectories = generate_trajectories(
        pde="heat1d",
        num_trajectories=8,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_1d,
        dt=cfg.dt,
        dx=cfg.dx,
        alpha=cfg.diffusion_coeff,
        seed=cfg.seed,
    )

    # Train autoencoder on clean data
    dataset_clean = PDEDataset(trajectories, mode="spatial", dim=1)
    model = build_autoencoder(
        dim=1,
        spatial_shape=(cfg.grid_size_1d,),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        spectral=False,
    )
    _train_model(model, dataset_clean, cfg)

    # Evaluate on noisy inputs
    errors = {}
    for std in cfg.noise_levels:
        noise_fn = gaussian_noise_fn(std=std)
        noisy_ds = PDEDataset(trajectories, mode="spatial", dim=1, noise_fn=noise_fn)
        mse = _evaluate_model(model, noisy_ds, cfg)
        errors[std] = mse
        print(f"Noise std {std:.4f} -> MSE {mse:.6f}")

    # Plot error vs noise level
    plt_path = os.path.join("plots", "noise_robustness.png")
    _plot_noise_errors(errors, plt_path)


def _train_model(model, dataset, cfg):
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
        print(f"[noise] Epoch {epoch}/{cfg.num_epochs_ae} loss {avg:.6f}")


def _evaluate_model(model, dataset, cfg):
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    device = torch.device(cfg.device)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    model = model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.ndim == 2:
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
            pred = model(x)
            total += loss_fn(pred, y).item()
            count += x.size(0)
    return total / count


def _plot_noise_errors(errors: dict[float, float], save_path: str):
    import matplotlib.pyplot as plt

    levels = sorted(errors.keys())
    values = [errors[k] for k in levels]
    plt.figure(figsize=(6, 4))
    plt.plot(levels, values, "o-")
    plt.xlabel("Noise std")
    plt.ylabel("MSE")
    plt.title("Autoencoder robustness to input noise")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    run_experiment()
