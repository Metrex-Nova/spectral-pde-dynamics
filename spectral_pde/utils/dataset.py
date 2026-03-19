"""PyTorch dataset for PDE trajectory data.

Provides a simple interface for sampling (u_t, u_{t+1}) pairs from simulated trajectories.
Supports both spatial and spectral representations, as well as 1D/2D data.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .fft_utils import to_spectral


class PDEDataset(Dataset):
    def __init__(
        self,
        trajectories: np.ndarray,
        mode: str = "spatial",
        dim: int = 1,
        noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Create a dataset of (u_t, u_{t+1}) pairs.

        Args:
            trajectories: numpy array of shape (N, T, ...).
            mode: 'spatial' or 'spectral'.
            dim: 1 or 2 (spatial dim of the data).
            noise_fn: optional function to add noise to inputs.
        """
        assert mode in ("spatial", "spectral"), "mode must be 'spatial' or 'spectral'"
        assert dim in (1, 2), "dim must be 1 or 2"

        self.trajectories = trajectories.astype(np.float32)
        self.mode = mode
        self.dim = dim
        self.noise_fn = noise_fn

        self.num_traj, self.time_steps = trajectories.shape[:2]
        self.spatial_shape = trajectories.shape[2:]
        self.length = self.num_traj * (self.time_steps - 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_idx = idx // (self.time_steps - 1)
        t = idx % (self.time_steps - 1)

        u_t = self.trajectories[traj_idx, t]
        u_next = self.trajectories[traj_idx, t + 1]

        x = torch.from_numpy(u_t).float()
        y = torch.from_numpy(u_next).float()

        if self.mode == "spectral":
            x = to_spectral(x, dim=self.dim)
            y = to_spectral(y, dim=self.dim)
            # move channel dimension first for 1D spectral data
            if self.dim == 1 and x.ndim == 2:
                x = np.transpose(x, (1, 0))
                y = np.transpose(y, (1, 0))

        if self.noise_fn is not None:
            x = self.noise_fn(x)

        return x, y

    def get_original_shape(self) -> Tuple[int, ...]:
        return self.spatial_shape
