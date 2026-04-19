from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from GOBC.models.dinov2_backbone import DinoV2Backbone


@dataclass(frozen=True)
class PatchMaskProjection:
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor


def project_mask_to_patch_grid(
    mask: torch.Tensor,
    grid_hw: tuple[int, int],
    threshold: float = 0.3,
) -> PatchMaskProjection:
    if mask.ndim != 4:
        raise ValueError(f"Expected [B, 1, H, W] masks, got {tuple(mask.shape)}")
    resized = F.interpolate(mask.float(), size=grid_hw, mode="area")
    soft_mask = resized.flatten(2).squeeze(1).clamp(0.0, 1.0)
    return PatchMaskProjection(
        hard_mask=soft_mask > threshold,
        soft_mask=soft_mask,
    )


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_soft_mask: torch.Tensor,
        context_hard_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~context_hard_mask,
            need_weights=False,
        )
        out = query + attn_out
        out = out + self.ffn(self.ffn_norm(out))
        return out * query_soft_mask.unsqueeze(-1)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    prob: torch.Tensor
    tokens1: torch.Tensor | None
    tokens2: torch.Tensor | None
    mask1: torch.Tensor | None
    mask2: torch.Tensor | None
    soft_mask1: torch.Tensor | None
    soft_mask2: torch.Tensor | None
    embedding1: torch.Tensor | None
    embedding2: torch.Tensor | None
    valid_indices: torch.Tensor


class PairwiseDifferenceModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module | None = None,
        backbone_source: str = "torchhub",
        backbone_name: str = "dinov2_vitb14",
        backbone_hf_name: str = "facebook/dinov2-base",
        freeze_backbone: bool = True,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask_threshold: float = 0.3,
        return_debug_tensors: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone or DinoV2Backbone(
            source=backbone_source,
            name=backbone_name,
            hf_name=backbone_hf_name,
            freeze=freeze_backbone,
        )
        self.mask_threshold = mask_threshold
        self.return_debug_tensors = return_debug_tensors
        self.proj = nn.LazyLinear(proj_dim)
        self.cross_12 = CrossAttentionBlock(proj_dim, num_heads, dropout)
        self.cross_21 = CrossAttentionBlock(proj_dim, num_heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(
        self,
        image1: torch.Tensor,
        mask1: torch.Tensor,
        image2: torch.Tensor,
        mask2: torch.Tensor,
    ) -> ModelOutput:
        tokens1, grid1 = self.backbone(image1)
        tokens2, grid2 = self.backbone(image2)
        patch_masks1 = project_mask_to_patch_grid(mask1, grid1, self.mask_threshold)
        patch_masks2 = project_mask_to_patch_grid(mask2, grid2, self.mask_threshold)
        valid = patch_masks1.hard_mask.any(dim=1) & patch_masks2.hard_mask.any(dim=1)

        if not valid.any():
            empty = tokens1.new_empty((0,))
            empty_tokens = tokens1.new_empty((0, tokens1.shape[1], self.proj.out_features))
            empty_hard_mask = patch_masks1.hard_mask.new_empty((0, patch_masks1.hard_mask.shape[1]))
            empty_soft_mask = tokens1.new_empty((0, patch_masks1.hard_mask.shape[1]))
            empty_embed = tokens1.new_empty((0, self.proj.out_features))
            return ModelOutput(
                logits=empty,
                prob=empty,
                tokens1=empty_tokens if self.return_debug_tensors else None,
                tokens2=empty_tokens if self.return_debug_tensors else None,
                mask1=empty_hard_mask if self.return_debug_tensors else None,
                mask2=empty_hard_mask if self.return_debug_tensors else None,
                soft_mask1=empty_soft_mask if self.return_debug_tensors else None,
                soft_mask2=empty_soft_mask if self.return_debug_tensors else None,
                embedding1=empty_embed if self.return_debug_tensors else None,
                embedding2=empty_embed if self.return_debug_tensors else None,
                valid_indices=valid.nonzero(as_tuple=False).squeeze(-1),
            )

        tokens1 = tokens1[valid]
        tokens2 = tokens2[valid]
        tokens1 = self.proj(tokens1)
        tokens2 = self.proj(tokens2)

        patch_mask1 = patch_masks1.hard_mask[valid]
        patch_mask2 = patch_masks2.hard_mask[valid]
        soft_mask1 = patch_masks1.soft_mask[valid]
        soft_mask2 = patch_masks2.soft_mask[valid]
        tokens1 = tokens1 * soft_mask1.unsqueeze(-1)
        tokens2 = tokens2 * soft_mask2.unsqueeze(-1)

        attended1 = self.cross_12(tokens1, tokens2, soft_mask1, patch_mask2)
        attended2 = self.cross_21(tokens2, tokens1, soft_mask2, patch_mask1)

        embedding1 = masked_mean_pool(attended1, soft_mask1)
        embedding2 = masked_mean_pool(attended2, soft_mask2)
        relation = torch.cat(
            [embedding1, embedding2, torch.abs(embedding1 - embedding2), embedding1 * embedding2],
            dim=-1,
        )
        logits = self.head(relation).squeeze(-1)
        return ModelOutput(
            logits=logits,
            prob=torch.sigmoid(logits),
            tokens1=attended1 if self.return_debug_tensors else None,
            tokens2=attended2 if self.return_debug_tensors else None,
            mask1=patch_mask1 if self.return_debug_tensors else None,
            mask2=patch_mask2 if self.return_debug_tensors else None,
            soft_mask1=soft_mask1 if self.return_debug_tensors else None,
            soft_mask2=soft_mask2 if self.return_debug_tensors else None,
            embedding1=embedding1 if self.return_debug_tensors else None,
            embedding2=embedding2 if self.return_debug_tensors else None,
            valid_indices=valid.nonzero(as_tuple=False).squeeze(-1),
        )
