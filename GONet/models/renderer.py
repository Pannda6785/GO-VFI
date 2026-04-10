"""Geometry-aware GO rendering and composition."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


MODE_STATIC = 0
MODE_APPEAR_DISAPPEAR = 1
MODE_APPEARANCE_CHANGE = 2
MODE_BLINK = 3
MODE_AFFINE = 4


class WarpRenderer(nn.Module):
    """Warp GO residuals to the midpoint according to predicted transforms."""

    def __init__(self, align_corners: bool = False) -> None:
        super().__init__()
        self.align_corners = align_corners

    def _warp_affine(self, source: torch.Tensor, affine_mats: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(affine_mats, source.shape, align_corners=self.align_corners)
        return F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=self.align_corners)

    def forward(
        self,
        g0: torch.Tensor,
        m0: torch.Tensor,
        mode_logits: torch.Tensor,
        transform_params: torch.Tensor,
        alpha: torch.Tensor,
        instance_valid: torch.Tensor,
        i_bg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render midpoint GO layers and compose a base image."""
        bsz, num_inst, channels, height, width = g0.shape

        flat_g0 = g0.reshape(bsz * num_inst, channels, height, width)
        flat_m0 = m0.reshape(bsz * num_inst, 1, height, width)
        affine_mats = transform_params.reshape(-1, 2, 3)

        warped_overlay = self._warp_affine(flat_g0, affine_mats).view(bsz, num_inst, channels, height, width)
        warped_mask = self._warp_affine(flat_m0, affine_mats).view(bsz, num_inst, 1, height, width)

        mode_ids = mode_logits.argmax(dim=-1)
        static_like = (
            (mode_ids == MODE_STATIC)
            | (mode_ids == MODE_APPEARANCE_CHANGE)
            | (mode_ids == MODE_AFFINE)
        ).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        blink = (mode_ids == MODE_BLINK).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        alpha_map = alpha.unsqueeze(-1).unsqueeze(-1)
        g_mid_per_instance = static_like * warped_overlay + blink * warped_overlay * alpha_map
        m_mid_per_instance = static_like * warped_mask + blink * warped_mask * alpha_map

        valid = instance_valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        g_mid_per_instance = g_mid_per_instance * valid
        m_mid_per_instance = m_mid_per_instance * valid

        g_mid = g_mid_per_instance.sum(dim=1)
        m_mid = m_mid_per_instance.sum(dim=1).clamp(0.0, 1.0)
        i_base = i_bg + g_mid

        return {
            "g_mid_per_instance": g_mid_per_instance,
            "m_mid_per_instance": m_mid_per_instance,
            "G_mid": g_mid,
            "M_mid": m_mid,
            "I_base": i_base,
        }

