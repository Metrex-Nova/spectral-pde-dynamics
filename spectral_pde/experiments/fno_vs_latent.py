"""Experiment: compare FNO vs latent dynamics models."""

from __future__ import annotations

import os

import numpy as np
import torch

from spectral_pde.config import get_config
from spectral_pde.data.pde_simulation import generate_trajectories
from spectral_pde.training.train_dynamics import train_latent_dynamics
from spectral_pde.training.train_fno import train_fno
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

    # train latent model
    ae, dynamics, _ = train_latent_dynamics(
        trajectories,
        (cfg.grid_size_1d,),
        cfg,
        mode="spatial",
        dim=1,
        use_lstm=False,
    )

    # train FNO model
    fno = train_fno(trajectories, (cfg.grid_size_1d,), cfg, dim=1)

    # Evaluate rollout on a single trajectory
    test_traj = trajectories[0]
    horizon = test_traj.shape[0] - 1
    device = torch.device(cfg.device)

    # latent rollout
    latent_rollout = np.zeros_like(test_traj)
    latent_rollout[0] = test_traj[0]
    z = ae.encode(torch.from_numpy(test_traj[0:1]).float().to(device))
    hidden = None
    for t in range(horizon):
        z, hidden = dynamics(z, hidden)
        u_pred = ae.decode(z).detach().cpu().numpy()[0]
        latent_rollout[t + 1] = u_pred

    # FNO rollout
    fno_rollout = np.zeros_like(test_traj)
    fno_rollout[0] = test_traj[0]
    fno = fno.to(device).eval()
    with torch.no_grad():
        for t in range(horizon):
            u = torch.from_numpy(fno_rollout[t:t+1]).float().to(device).unsqueeze(1)
            u_pred = fno(u).squeeze(1).cpu().numpy()
            fno_rollout[t + 1] = u_pred[0]

    # Plot error curves
    plot_error_over_time(
        true=test_traj,
        pred=latent_rollout,
        title="Latent dynamics rollout error",
        save_path=os.path.join("plots", "latent_rollout_error.png"),
    )

    plot_error_over_time(
        true=test_traj,
        pred=fno_rollout,
        title="FNO rollout error",
        save_path=os.path.join("plots", "fno_rollout_error.png"),
    )


if __name__ == "__main__":
    run_experiment()
