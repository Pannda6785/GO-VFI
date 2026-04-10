"""Shared feature encoder for overlay-centric GO reasoning."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Small convolutional block used by the siamese encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiameseEncoder(nn.Module):
    """Encode per-instance overlay tensors with shared weights."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, hidden_dim, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stage3(self.stage2(self.stem(x)))
        return self.pool(features).flatten(1)

