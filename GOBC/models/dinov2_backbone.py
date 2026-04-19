from __future__ import annotations

import importlib
import math
from typing import Any

import torch
from torch import nn


DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def infer_patch_grid(
    image_hw: tuple[int, int] | None,
    num_tokens: int,
    patch_size: int | None,
) -> tuple[int, int]:
    if image_hw is not None and patch_size:
        height, width = image_hw
        if height % patch_size == 0 and width % patch_size == 0:
            grid = (height // patch_size, width // patch_size)
            if grid[0] * grid[1] == num_tokens:
                return grid

    side = int(math.isqrt(num_tokens))
    if side * side == num_tokens:
        return side, side

    for gh in range(side, 0, -1):
        if num_tokens % gh == 0:
            return gh, num_tokens // gh

    raise ValueError(f"Could not infer patch grid from {num_tokens} tokens.")


def _as_patch_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and value:
        return int(value[0])
    return None


class DinoV2Backbone(nn.Module):
    """Thin wrapper over DINOv2 with a stable patch-token API."""

    def __init__(
        self,
        source: str = "torchhub",
        name: str = "dinov2_vitb14",
        hf_name: str = "facebook/dinov2-base",
        freeze: bool = True,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.name = name
        self.hf_name = hf_name
        self.model = model if model is not None else self._load_model()
        self.patch_size = self._infer_patch_size()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _load_model(self) -> nn.Module:
        if self.source == "torchhub":
            return torch.hub.load("facebookresearch/dinov2", self.name)
        if self.source == "hf":
            transformers = importlib.import_module("transformers")
            return transformers.Dinov2Model.from_pretrained(self.hf_name)
        raise ValueError(f"Unsupported DINOv2 source: {self.source}")

    def _infer_patch_size(self) -> int | None:
        if hasattr(self.model, "patch_embed"):
            return _as_patch_size(getattr(self.model.patch_embed, "patch_size", None))
        config = getattr(self.model, "config", None)
        if config is not None:
            return _as_patch_size(getattr(config, "patch_size", None))
        return None

    def _extract_torchhub_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(images)
        else:
            features = self.model(images)

        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"]
            if "patch_tokens" in features:
                return features["patch_tokens"]
            if "last_hidden_state" in features:
                return features["last_hidden_state"][:, 1:, :]
            if "x_prenorm" in features:
                start = 1 + int(getattr(self.model, "num_register_tokens", 0))
                return features["x_prenorm"][:, start:, :]
            raise KeyError(f"Unsupported torchhub feature keys: {sorted(features)}")

        if isinstance(features, torch.Tensor):
            if features.ndim != 3:
                raise ValueError(f"Expected [B, N, C] features, got {tuple(features.shape)}")
            return features[:, 1:, :] if features.shape[1] > 1 else features

        raise TypeError(f"Unsupported torchhub feature type: {type(features)!r}")

    def _extract_hf_tokens(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise KeyError("Hugging Face Dinov2Model did not return last_hidden_state.")
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError(f"Unexpected HF hidden state shape: {tuple(hidden.shape)}")
        return hidden[:, 1:, :]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, 3, H, W] images, got {tuple(images.shape)}")

        if self.source == "torchhub":
            tokens = self._extract_torchhub_tokens(images)
        elif self.source == "hf":
            tokens = self._extract_hf_tokens(images)
        else:
            raise ValueError(f"Unsupported DINOv2 source: {self.source}")

        grid_hw = infer_patch_grid(images.shape[-2:], tokens.shape[1], self.patch_size)
        return tokens, grid_hw

