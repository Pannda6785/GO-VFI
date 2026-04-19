from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from GOBC.data.pair_dataset import VimeoOverlayPairDataset, build_pair_index, derive_overlay_label
from GOBC.models.cross_attention_matcher import PairwiseDifferenceModel, project_mask_to_patch_grid
from GOBC.models.dinov2_backbone import DinoV2Backbone, infer_patch_grid


class FakeTorchHubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = types.SimpleNamespace(patch_size=14)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        grid_h = images.shape[-2] // 14
        grid_w = images.shape[-1] // 14
        tokens = torch.randn(batch, grid_h * grid_w, 32, device=images.device)
        return {"x_norm_patchtokens": tokens}


class FakeHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(patch_size=14)

    def forward(self, pixel_values: torch.Tensor) -> types.SimpleNamespace:
        batch = pixel_values.shape[0]
        grid_h = pixel_values.shape[-2] // 14
        grid_w = pixel_values.shape[-1] // 14
        cls = torch.randn(batch, 1, 48, device=pixel_values.device)
        patches = torch.randn(batch, grid_h * grid_w, 48, device=pixel_values.device)
        return types.SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))


class DummyBackbone(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch = images.shape[0]
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))
        tokens = pooled.flatten(2).transpose(1, 2)
        repeats = (64 + tokens.shape[-1] - 1) // tokens.shape[-1]
        tokens = tokens.repeat(1, 1, repeats)[:, :, :64]
        return tokens.reshape(batch, 16, 64), (4, 4)


def write_png(path: Path, value: int) -> None:
    image = Image.new("L", (32, 32), color=value)
    image.save(path)


def write_rgb(path: Path, value: int) -> None:
    image = Image.new("RGB", (32, 32), color=(value, value, value))
    image.save(path)


class TestGOBC(unittest.TestCase):
    def test_infer_patch_grid(self) -> None:
        self.assertEqual(infer_patch_grid((518, 518), 37 * 37, 14), (37, 37))
        self.assertEqual(infer_patch_grid(None, 16, None), (4, 4))

    def test_backbone_output_shape_torchhub(self) -> None:
        with mock.patch("torch.hub.load", return_value=FakeTorchHubModel()):
            backbone = DinoV2Backbone(source="torchhub")
            tokens, grid = backbone(torch.randn(2, 3, 518, 518))
        self.assertEqual(tokens.shape, (2, 37 * 37, 32))
        self.assertEqual(grid, (37, 37))

    def test_backbone_output_shape_hf(self) -> None:
        fake_transformers = types.SimpleNamespace(
            Dinov2Model=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=FakeHFModel()))
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            backbone = DinoV2Backbone(source="hf")
            tokens, grid = backbone(torch.randn(2, 3, 518, 518))
        self.assertEqual(tokens.shape, (2, 37 * 37, 48))
        self.assertEqual(grid, (37, 37))

    def test_mask_projection(self) -> None:
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 3:7] = 1.0
        projected = project_mask_to_patch_grid(mask, (4, 4), threshold=0.5)
        self.assertEqual(projected.shape, (1, 16))
        self.assertGreater(projected.sum().item(), 0)

    def test_model_forward_shape(self) -> None:
        model = PairwiseDifferenceModel(backbone=DummyBackbone(), proj_dim=32, num_heads=4)
        batch = {
            "image1": torch.randn(3, 3, 64, 64),
            "image2": torch.randn(3, 3, 64, 64),
            "mask1": torch.ones(3, 1, 64, 64),
            "mask2": torch.ones(3, 1, 64, 64),
        }
        out = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
        self.assertEqual(out.logits.shape, (3,))
        self.assertEqual(out.prob.shape, (3,))
        self.assertEqual(out.mask1.shape, (3, 16))

    def test_training_step_backward(self) -> None:
        model = PairwiseDifferenceModel(backbone=DummyBackbone(), proj_dim=32, num_heads=4)
        output = model(
            torch.randn(2, 3, 64, 64),
            torch.ones(2, 1, 64, 64),
            torch.randn(2, 3, 64, 64),
            torch.ones(2, 1, 64, 64),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, torch.tensor([0.0, 1.0]))
        loss.backward()
        has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        self.assertTrue(has_grad)

    def test_checkpoint_save_load_consistency(self) -> None:
        model_a = PairwiseDifferenceModel(backbone=DummyBackbone(), proj_dim=32, num_heads=4)
        model_b = PairwiseDifferenceModel(backbone=DummyBackbone(), proj_dim=32, num_heads=4)
        batch = (
            torch.randn(2, 3, 64, 64),
            torch.ones(2, 1, 64, 64),
            torch.randn(2, 3, 64, 64),
            torch.ones(2, 1, 64, 64),
        )
        _ = model_a(*batch)
        _ = model_b(*batch)
        model_b.load_state_dict(model_a.state_dict())
        out_a = model_a(*batch)
        out_b = model_b(*batch)
        self.assertTrue(torch.allclose(out_a.logits, out_b.logits))

    def test_dataset_index_and_label_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "train" / "00001" / "0001"
            (root / "overlays_masks").mkdir(parents=True)
            write_rgb(root / "I0.png", 120)
            write_rgb(root / "I1.png", 140)
            write_png(root / "overlays_masks" / "000_I0.png", 0)
            write_png(root / "overlays_masks" / "000_I1.png", 0)
            write_png(root / "overlays_masks" / "001_I0.png", 255)
            write_png(root / "overlays_masks" / "001_I1.png", 255)
            metadata = {
                "split": "train",
                "source_rel": "00001/0001",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "000",
                        "temporal": {"mode": "appear_disappear", "detail": {"variant": "visibility"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    },
                    {
                        "object_id": "001",
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "mask_paths": {"I0": "overlays_masks/001_I0.png", "I1": "overlays_masks/001_I1.png"},
                    },
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            index = build_pair_index(Path(tmpdir) / "dataset", "train")
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0].label, 1)
            dataset = VimeoOverlayPairDataset(Path(tmpdir) / "dataset", "train", image_size=28)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image1"].shape), (3, 28, 28))
            self.assertEqual(tuple(sample["mask1"].shape), (1, 28, 28))
            self.assertEqual(sample["label"].item(), 1.0)

    def test_scenechange_sequence_is_dropped(self) -> None:
        overlay = {"temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}}}
        self.assertEqual(derive_overlay_label(overlay), 1)


if __name__ == "__main__":
    unittest.main()
