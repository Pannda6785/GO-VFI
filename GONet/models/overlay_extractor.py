"""Overlay extraction utilities for GO handling."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class OverlayExtractor(nn.Module):
    """Compute overlay-only residual layers and masked inputs per GO instance.

    Expected batch fields:
        I0, I1, I0_bg, I1_bg: (B, 3, H, W)
        M0, M1: (B, N, 1, H, W)
        instance_valid: (B, N)
    """

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return per-instance overlay residuals and helper masks."""
        i0 = batch["I0"]
        i1 = batch["I1"]
        i0_bg = batch["I0_bg"]
        i1_bg = batch["I1_bg"]
        m0 = batch["M0"]
        m1 = batch["M1"]
        valid = batch["instance_valid"].float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        i0_residual = (i0 - i0_bg).unsqueeze(1)
        i1_residual = (i1 - i1_bg).unsqueeze(1)

        g0 = i0_residual * m0 * valid
        g1 = i1_residual * m1 * valid

        return {
            "G0": g0,
            "G1": g1,
            "M0": m0 * valid,
            "M1": m1 * valid,
            "instance_valid": batch["instance_valid"].float(),
        }

