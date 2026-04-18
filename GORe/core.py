from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand_as(pred)
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(eps)
    return diff.sum() / denom


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = masked_l1(dx_pred, dx_tgt, mask[:, :, :, 1:], eps=eps)
    loss_y = masked_l1(dy_pred, dy_tgt, mask[:, :, 1:, :], eps=eps)
    return loss_x + loss_y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.down(feat)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


@dataclass(frozen=True)
class ModelConfig:
    stage_channels: tuple[int, ...]
    bottleneck_channels: int


class OverlayRefiner(nn.Module):
    """
    Local overlay refinement network.

    Inputs:
    - I0
    - I0.5_cp
    - M
    - E
    - T

    The model predicts a residual RGB correction delta. Refined content is:
        P = I0.5_cp + delta

    Final output is composed as:
        I_hat = (1 - E) * I0.5_cp + E * P
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        in_ch = 9
        self.stage_channels = config.stage_channels

        downs = []
        prev_ch = in_ch
        for out_ch in self.stage_channels:
            downs.append(DownBlock(prev_ch, out_ch))
            prev_ch = out_ch
        self.downs = nn.ModuleList(downs)

        self.bottleneck = ConvBlock(self.stage_channels[-1], config.bottleneck_channels)

        ups = []
        in_ch = config.bottleneck_channels
        for skip_ch in reversed(self.stage_channels):
            ups.append(UpBlock(in_ch, skip_ch, skip_ch))
            in_ch = skip_ch
        self.ups = nn.ModuleList(ups)

        final_ch = self.stage_channels[0]
        self.out_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_ch, 3, kernel_size=3, padding=1),
        )

    def forward(
        self,
        i0: torch.Tensor,
        i05_cp: torch.Tensor,
        m: torch.Tensor,
        e: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([i0, i05_cp, m, e, t], dim=1)
        skips = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        delta = self.out_head(x)
        pred_patch = i05_cp + delta
        pred = (1.0 - e) * i05_cp + e * pred_patch
        return {"delta": delta, "pred_patch": pred_patch, "pred": pred}


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(stage_channels=(32, 64, 128), bottleneck_channels=256),
    "8m": ModelConfig(stage_channels=(32, 64, 128, 256), bottleneck_channels=448),
}


def build_model(variant: str = "small") -> OverlayRefiner:
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model variant: {variant}")
    return OverlayRefiner(MODEL_CONFIGS[variant])


class OverlayRefinerLoss(nn.Module):
    def __init__(self, lambda_e: float = 1.0, beta_grad: float = 0.5):
        super().__init__()
        self.lambda_e = lambda_e
        self.beta_grad = beta_grad

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        e: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_global = torch.mean(torch.abs(pred - target))
        loss_e = masked_l1(pred, target, e)
        loss_grad = gradient_loss(pred, target, t)
        loss = loss_global + self.lambda_e * loss_e + self.beta_grad * loss_grad
        return {
            "loss": loss,
            "loss_global": loss_global.detach(),
            "loss_e": loss_e.detach(),
            "loss_grad": loss_grad.detach(),
        }


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    use_amp: bool
    amp_device_type: str | None
    amp_dtype: torch.dtype | None


def make_runtime() -> Runtime:
    if torch.cuda.is_available():
        return Runtime(torch.device("cuda"), True, "cuda", torch.bfloat16)
    if torch.backends.mps.is_available():
        return Runtime(torch.device("mps"), False, None, None)
    return Runtime(torch.device("cpu"), False, None, None)


def autocast_context(runtime: Runtime):
    if runtime.use_amp and runtime.amp_device_type is not None and runtime.amp_dtype is not None:
        return torch.autocast(device_type=runtime.amp_device_type, dtype=runtime.amp_dtype)
    return nullcontext()
