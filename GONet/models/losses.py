"""Loss stubs for GONet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class GOHandlerLoss(nn.Module):
    """Compute task losses for GO handler training."""

    def __init__(
        self,
        mode_weight: float = 1.0,
        transform_weight: float = 1.0,
        go_mid_weight: float = 1.0,
        final_frame_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode_weight = mode_weight
        self.transform_weight = transform_weight
        self.go_mid_weight = go_mid_weight
        self.final_frame_weight = final_frame_weight

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return individual and total loss values."""
        device = outputs["I_final"].device
        losses: Dict[str, torch.Tensor] = {}
        instance_valid = batch["instance_valid"].float()
        valid_count = instance_valid.sum().clamp_min(1.0)

        if "mode_gt" in batch:
            mode_logits = outputs["mode_logits"].reshape(-1, outputs["mode_logits"].shape[-1])
            mode_gt = batch["mode_gt"].reshape(-1)
            valid_flat = instance_valid.reshape(-1) > 0
            losses["mode_loss"] = (
                F.cross_entropy(mode_logits[valid_flat], mode_gt[valid_flat])
                if valid_flat.any()
                else torch.zeros((), device=device)
            )
        else:
            losses["mode_loss"] = torch.zeros((), device=device)

        if "transform_gt" in batch:
            diff = torch.abs(outputs["transform_params"] - batch["transform_gt"])
            losses["transform_loss"] = (diff * instance_valid.unsqueeze(-1)).sum() / valid_count
        else:
            losses["transform_loss"] = torch.zeros((), device=device)

        if "G_mid_gt" in batch:
            target = batch["G_mid_gt"]
            if target.dim() == 5:
                target = target.sum(dim=1)
            losses["go_mid_loss"] = F.l1_loss(outputs["G_mid"], target)
        else:
            losses["go_mid_loss"] = torch.zeros((), device=device)

        if "I_mid_gt" in batch:
            losses["final_frame_loss"] = F.l1_loss(outputs["I_final"], batch["I_mid_gt"])
        else:
            losses["final_frame_loss"] = torch.zeros((), device=device)

        losses["total_loss"] = (
            self.mode_weight * losses["mode_loss"]
            + self.transform_weight * losses["transform_loss"]
            + self.go_mid_weight * losses["go_mid_loss"]
            + self.final_frame_weight * losses["final_frame_loss"]
        )
        return losses

