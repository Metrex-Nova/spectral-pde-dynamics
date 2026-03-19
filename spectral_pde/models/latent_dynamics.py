"""Latent-space dynamics models (MLP, optional LSTM)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    """Predict z_{t+1} from z_t in latent space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        use_lstm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm

        if use_lstm:
            self.lstm = nn.LSTMCell(latent_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, latent_dim)
        else:
            self.model = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(
        self,
        z: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Predict the next latent state.

        Args:
            z: tensor of shape (batch, latent_dim)
            hidden: optional (h, c) for LSTM.

        Returns:
            z_next: tensor of shape (batch, latent_dim)
            hidden: updated LSTM hidden state (None if not using LSTM)
        """
        if self.use_lstm:
            if hidden is None:
                h = z.new_zeros(z.size(0), self.hidden_dim)
                c = z.new_zeros(z.size(0), self.hidden_dim)
                hidden = (h, c)
            h, c = self.lstm(z, hidden)
            z_next = self.fc(h)
            return z_next, (h, c)

        return self.model(z), None
