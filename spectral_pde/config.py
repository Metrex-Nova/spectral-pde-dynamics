"""Configuration for spectral PDE learning project."""

from dataclasses import dataclass


@dataclass
class Config:
    # Simulation settings
    spatial_dim: int = 1  # 1 or 2
    grid_size_1d: int = 64
    grid_size_2d: int = 32
    time_steps: int = 20
    dt: float = 0.01
    dx: float = 1.0

    # PDE parameters
    diffusion_coeff: float = 0.1  # alpha for heat equation
    wave_speed: float = 1.0  # c for wave equation

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs_ae: int = 5
    num_epochs_dynamics: int = 5
    num_epochs_fno: int = 5

    latent_dim: int = 32
    hidden_dim: int = 128

    # Noise
    noise_std: float = 0.01
    noise_levels: list[float] = (0.0, 0.01, 0.05, 0.1)

    # FNO
    fno_modes_1d: int = 8
    fno_modes_2d: int = 8
    fno_width: int = 32
    fno_layers: int = 3

    # Misc
    device: str = "cpu"
    seed: int = 42


def get_config() -> Config:
    return Config()
