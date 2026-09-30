"""Task termination from the frozen VLM's final valid input-prefix feature."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def last_prefix_features(hidden: Tensor, valid: Tensor) -> Tensor:
    """Copy only [B,D], excluding padding and teacher-forced answer positions."""
    if (hidden.ndim != 3 or valid.shape != hidden.shape[:2]
            or valid.dtype != torch.bool or not valid.any(dim=1).all()):
        raise ValueError("stop features require hidden [B,N,D] and nonempty bool prefix [B,N]")
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
    last = positions.masked_fill(~valid, -1).amax(dim=1)
    # Advanced indexing owns its storage: no [B,N,D] allocation stays alive.
    return hidden.detach()[torch.arange(hidden.shape[0], device=hidden.device), last]


class LastPrefixStopHead(nn.Module):
    """An independent FP32 MLP; neither STOP gradients nor inputs pass through FM."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        # Keep the small classifier numerically stable even inside BF16 training.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.detach().to(dtype=self.layers[0].weight.dtype)
            return self.layers(features).squeeze(-1)
