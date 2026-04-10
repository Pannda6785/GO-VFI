"""Temporal interpretation for graphical overlays."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .encoder import SiameseEncoder


class TemporalInterpreter(nn.Module):
    """Predict mode, midpoint transform, and alpha from per-instance overlays."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        num_modes: int = 5,
        transform_dim: int = 6,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.transform_dim = transform_dim
        self.encoder = SiameseEncoder(in_channels=in_channels, hidden_dim=hidden_dim)

        joint_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mode_head = nn.Linear(hidden_dim, num_modes)
        self.transform_head = nn.Linear(hidden_dim, transform_dim)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self._init_transform_head()

    def _init_transform_head(self) -> None:
        """Initialize transform prediction near identity affine parameters."""
        nn.init.zeros_(self.transform_head.weight)
        identity_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        with torch.no_grad():
            self.transform_head.bias.copy_(identity_bias[: self.transform_dim])

    def forward(
        self,
        g0: torch.Tensor,
        g1: torch.Tensor,
        m0: torch.Tensor,
        m1: torch.Tensor,
        instance_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run per-instance temporal interpretation."""
        bsz, num_inst, _, height, width = g0.shape

        x0 = torch.cat([g0, m0], dim=2).reshape(bsz * num_inst, 4, height, width)
        x1 = torch.cat([g1, m1], dim=2).reshape(bsz * num_inst, 4, height, width)

        feat0 = self.encoder(x0)
        feat1 = self.encoder(x1)
        fused = self.fusion(torch.cat([feat0, feat1, feat1 - feat0], dim=1))

        mode_logits = self.mode_head(fused).view(bsz, num_inst, self.num_modes)
        transform_params = self.transform_head(fused).view(bsz, num_inst, self.transform_dim)
        alpha = self.alpha_head(fused).view(bsz, num_inst, 1)

        valid = instance_valid.unsqueeze(-1)
        return {
            "mode_logits": mode_logits * valid,
            "transform_params": transform_params * valid,
            "alpha": alpha * valid,
        }

