"""Full GONet wiring."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .interpreter import TemporalInterpreter
from .overlay_extractor import OverlayExtractor
from .refiner import EdgeRefiner
from .renderer import WarpRenderer


class GOHandlerNet(nn.Module):
    """Graphical overlay handler with interpreter, renderer, and refiner."""

    def __init__(
        self,
        num_modes: int = 5,
        transform_dim: int = 6,
        encoder_in_channels: int = 4,
        encoder_hidden_dim: int = 128,
        refiner_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.overlay_extractor = OverlayExtractor()
        self.interpreter = TemporalInterpreter(
            in_channels=encoder_in_channels,
            hidden_dim=encoder_hidden_dim,
            num_modes=num_modes,
            transform_dim=transform_dim,
        )
        self.renderer = WarpRenderer()
        self.refiner = EdgeRefiner(hidden_dim=refiner_hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the full GO restoration branch."""
        overlay_outputs = self.overlay_extractor(batch)
        interpreter_outputs = self.interpreter(
            overlay_outputs["G0"],
            overlay_outputs["G1"],
            overlay_outputs["M0"],
            overlay_outputs["M1"],
            overlay_outputs["instance_valid"],
        )
        renderer_outputs = self.renderer(
            overlay_outputs["G0"],
            overlay_outputs["M0"],
            interpreter_outputs["mode_logits"],
            interpreter_outputs["transform_params"],
            interpreter_outputs["alpha"],
            overlay_outputs["instance_valid"],
            batch["I_bg"],
        )
        refiner_outputs = self.refiner(
            renderer_outputs["I_base"],
            batch["I_bg"],
            renderer_outputs["G_mid"],
            renderer_outputs["M_mid"],
        )

        return {
            **overlay_outputs,
            **interpreter_outputs,
            **renderer_outputs,
            **refiner_outputs,
            "I_final": refiner_outputs["I_final"],
        }

