"""Main entry point for the spectral PDE learning project.

This script runs a full pipeline:
- generate datasets for 1D/2D PDEs
- train autoencoders (spatial and spectral)
- train latent dynamics models
- train Fourier Neural Operator (FNO) models
- run comparison experiments and save plots
"""

from __future__ import annotations

import os

from spectral_pde.config import get_config
from spectral_pde.data.pde_simulation import generate_trajectories
from spectral_pde.training.train_autoencoder import train_autoencoder
from spectral_pde.training.train_dynamics import train_latent_dynamics
from spectral_pde.training.train_fno import train_fno
from spectral_pde.experiments.compare_spatial_vs_spectral import run_experiment as run_compare
from spectral_pde.experiments.noise_robustness import run_experiment as run_noise
from spectral_pde.experiments.fno_vs_latent import run_experiment as run_fno_vs_latent


def main() -> None:
    cfg = get_config()
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Generate datasets
    print("Generating datasets...")
    heat1d = generate_trajectories(
        pde="heat1d",
        num_trajectories=16,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_1d,
        dt=cfg.dt,
        dx=cfg.dx,
        alpha=cfg.diffusion_coeff,
        seed=cfg.seed,
    )

    wave1d = generate_trajectories(
        pde="wave1d",
        num_trajectories=16,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_1d,
        dt=cfg.dt,
        dx=cfg.dx,
        c=cfg.wave_speed,
        seed=cfg.seed + 1,
    )

    heat2d = generate_trajectories(
        pde="heat2d",
        num_trajectories=8,
        time_steps=cfg.time_steps,
        grid_size=cfg.grid_size_2d,
        dt=cfg.dt,
        dx=cfg.dx,
        alpha=cfg.diffusion_coeff,
        seed=cfg.seed + 2,
    )

    print("Training autoencoders (1D spatial & spectral)...")
    _ = train_autoencoder(heat1d, (cfg.grid_size_1d,), cfg, mode="spatial", dim=1)
    _ = train_autoencoder(heat1d, (cfg.grid_size_1d,), cfg, mode="spectral", dim=1)

    print("Training latent dynamics (1D, spatial)...")
    _ = train_latent_dynamics(heat1d, (cfg.grid_size_1d,), cfg, mode="spatial", dim=1)

    print("Training FNO (1D heat)...")
    _ = train_fno(heat1d, (cfg.grid_size_1d,), cfg, dim=1)

    print("Training FNO (2D heat)...")
    _ = train_fno(heat2d, (cfg.grid_size_2d, cfg.grid_size_2d), cfg, dim=2)

    print("Running experiments...")
    run_compare()
    run_noise()
    run_fno_vs_latent()

    print("Done. Plots saved in ./plots and checkpoints in ./checkpoints.")


if __name__ == "__main__":
    main()
