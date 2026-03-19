"""PDE simulation utilities for 1D/2D heat and wave equations.

The goal is to generate training data for learning dynamics. We provide simple finite
difference solvers with periodic boundary conditions and random smooth initial states.

The output trajectories are stored as numpy arrays, suitable for conversion to torch tensors.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple, List


def _smooth_random_field_1d(n: int, sigma: float = 5.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth 1D random field via low-pass filtering in Fourier domain."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    fft = np.fft.rfft(x)
    k = np.fft.rfftfreq(n)
    filter_mask = np.exp(-0.5 * (k * n / sigma) ** 2)
    field = np.fft.irfft(fft * filter_mask, n=n)
    # normalize to unit variance
    field = (field - field.mean()) / (field.std() + 1e-10)
    return field


def _smooth_random_field_2d(n: int, sigma: float = 4.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth 2D random field via low-pass filtering in Fourier domain."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    fft = np.fft.rfft2(x)
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.rfftfreq(n)[None, :]
    k2 = kx**2 + ky**2
    filter_mask = np.exp(-0.5 * (np.sqrt(k2) * n / sigma) ** 2)
    field = np.fft.irfft2(fft * filter_mask, s=(n, n))
    field = (field - field.mean()) / (field.std() + 1e-10)
    return field


def simulate_heat_1d(
    initial: np.ndarray,
    time_steps: int,
    dt: float,
    dx: float,
    alpha: float = 0.1,
) -> np.ndarray:
    """Simulate the 1D heat equation u_t = alpha * u_xx with periodic BC.

    Args:
        initial: (N,) array of initial condition.
        time_steps: Number of time steps to simulate.
        dt: Time step size.
        dx: Spatial grid spacing.
        alpha: Diffusion coefficient.

    Returns:
        trajectory: (time_steps, N) array.
    """

    u = initial.copy().astype(np.float64)
    N = u.shape[0]
    trajectory = np.zeros((time_steps, N), dtype=np.float32)
    trajectory[0] = u

    # explicit finite difference scheme
    coeff = alpha * dt / (dx**2)
    for t in range(1, time_steps):
        u_xx = np.roll(u, -1) - 2 * u + np.roll(u, 1)
        u = u + coeff * u_xx
        trajectory[t] = u

    return trajectory


def simulate_wave_1d(
    initial: np.ndarray,
    time_steps: int,
    dt: float,
    dx: float,
    c: float = 1.0,
) -> np.ndarray:
    """Simulate the 1D wave equation u_tt = c^2 u_xx with periodic BC.

    Uses a standard second-order finite difference in time and space.
    """

    u0 = initial.copy().astype(np.float64)
    N = u0.shape[0]
    trajectory = np.zeros((time_steps, N), dtype=np.float32)
    trajectory[0] = u0

    # assume zero initial velocity
    u1 = u0.copy()
    c1 = (c * dt / dx) ** 2

    # first step: use u_tt approx (u1 - u0)/dt ~ 0 -> u1 = u0
    # so we already have u1
    if time_steps > 1:
        u = u0.copy()
        for t in range(1, time_steps):
            u_next = 2 * u1 - u + c1 * (np.roll(u1, -1) - 2 * u1 + np.roll(u1, 1))
            trajectory[t] = u_next
            u, u1 = u1, u_next

    return trajectory


def simulate_heat_2d(
    initial: np.ndarray,
    time_steps: int,
    dt: float,
    dx: float,
    alpha: float = 0.1,
) -> np.ndarray:
    """Simulate the 2D heat equation u_t = alpha * (u_xx + u_yy) with periodic BC."""

    u = initial.copy().astype(np.float64)
    N = u.shape[0]
    trajectory = np.zeros((time_steps, N, N), dtype=np.float32)
    trajectory[0] = u

    coeff = alpha * dt / (dx**2)
    for t in range(1, time_steps):
        u_xx = np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)
        u_yy = np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)
        u = u + coeff * (u_xx + u_yy)
        trajectory[t] = u

    return trajectory


def generate_trajectories(
    pde: str,
    num_trajectories: int,
    time_steps: int,
    grid_size: int,
    dt: float,
    dx: float,
    alpha: float = 0.1,
    c: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate multiple trajectories for a given PDE.

    Args:
        pde: 'heat1d', 'wave1d', 'heat2d'
        num_trajectories: number of independent runs.
        time_steps: number of time steps in each trajectory.
        grid_size: spatial size (N for 1D, N for 2D grid).
        dt: time step.
        dx: spatial step.
        alpha: diffusion coefficient (heat).
        c: wave speed.

    Returns:
        trajectories: numpy array of shape (num_trajectories, time_steps, ...)
    """

    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for i in range(num_trajectories):
        if pde == "heat1d":
            init = _smooth_random_field_1d(grid_size, sigma=8.0, seed=rng.integers(1_000_000))
            traj = simulate_heat_1d(init, time_steps, dt, dx, alpha)
        elif pde == "wave1d":
            init = _smooth_random_field_1d(grid_size, sigma=8.0, seed=rng.integers(1_000_000))
            traj = simulate_wave_1d(init, time_steps, dt, dx, c)
        elif pde == "heat2d":
            init = _smooth_random_field_2d(grid_size, sigma=4.0, seed=rng.integers(1_000_000))
            traj = simulate_heat_2d(init, time_steps, dt, dx, alpha)
        else:
            raise ValueError(f"Unknown PDE type: {pde}")
        out.append(traj)
    return np.stack(out, axis=0)
