"""Local boundary refinement for composed GO images."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class EdgeRefiner(nn.Module):
    """Refine composed GO output only around mask boundaries."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1),
        )

    @staticmethod
    def compute_edge_prior(mask: torch.Tensor) -> torch.Tensor:
        """Compute a simple boundary ring from midpoint masks."""
        dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        eroded = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(
        self,
        i_base: torch.Tensor,
        i_bg: torch.Tensor,
        g_mid: torch.Tensor,
        m_mid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Refine the composed image using local boundary edits."""
        edge_prior = self.compute_edge_prior(m_mid)
        delta = self.net(torch.cat([i_base, i_bg, g_mid, edge_prior], dim=1))
        delta = delta * edge_prior
        return {
            "edge_prior": edge_prior,
            "refiner_delta": delta,
            "I_final": i_base + delta,
        }

