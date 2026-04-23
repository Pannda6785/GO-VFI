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

from GOBC.data.pair_dataset import VimeoOverlayPairDataset, _overlay_has_motion, build_pair_index, derive_overlay_label
from GOBC.models.cross_attention_matcher import PairwiseDifferenceModel, project_mask_to_patch_grid
from GOBC.models.dinov2_backbone import DinoV2Backbone, infer_patch_grid
from GOBC.train import prepare_epoch_subsets, resolve_pos_weight_value


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


class DummyDataset:
    def __init__(self, split: str, valid_count: int, total_count: int) -> None:
        self.split = split
        self._valid_count = valid_count
        self._total_count = total_count

    def __len__(self) -> int:
        return self._total_count

    def is_valid_index(self, index: int) -> bool:
        return index < self._valid_count


class DummyLabeledDataset:
    def __init__(self, labels: list[int], valid_indices: set[int]) -> None:
        self.samples = [types.SimpleNamespace(label=label) for label in labels]
        self._valid_indices = valid_indices

    def is_valid_index(self, index: int) -> bool:
        return index in self._valid_indices


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
        projected = project_mask_to_patch_grid(mask, (4, 4), threshold=0.3)
        self.assertEqual(projected.hard_mask.shape, (1, 16))
        self.assertEqual(projected.soft_mask.shape, (1, 16))
        self.assertEqual(projected.hard_mask.dtype, torch.bool)
        self.assertTrue(projected.soft_mask.dtype.is_floating_point)
        self.assertGreater(projected.hard_mask.sum().item(), 0)
        self.assertGreater(projected.soft_mask.sum().item(), 0.0)
        self.assertTrue(torch.equal(projected.hard_mask, projected.soft_mask > 0.3))

    def test_model_forward_shape(self) -> None:
        model = PairwiseDifferenceModel(backbone=DummyBackbone(), proj_dim=32, num_heads=4, return_debug_tensors=True)
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
        self.assertEqual(out.soft_mask1.shape, (3, 16))
        self.assertEqual(out.mask1.dtype, torch.bool)
        self.assertTrue(out.soft_mask1.dtype.is_floating_point)

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
                        "center0": [8.0, 8.0],
                        "center1": [8.0, 8.0],
                        "temporal": {"mode": "appear_disappear", "detail": {"variant": "visibility"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    },
                    {
                        "object_id": "001",
                        "center0": [16.0, 16.0],
                        "center1": [16.0, 16.0],
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

    def test_overlay_with_off_frame_center_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "train" / "00001" / "0001"
            (root / "overlays_masks").mkdir(parents=True)
            write_rgb(root / "I0.png", 100)
            write_rgb(root / "I1.png", 120)
            write_png(root / "overlays_masks" / "000_I0.png", 255)
            write_png(root / "overlays_masks" / "000_I1.png", 255)
            metadata = {
                "split": "train",
                "source_rel": "00001/0001",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "000",
                        "center0": [40.0, 16.0],
                        "center1": [16.0, 16.0],
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            index = build_pair_index(Path(tmpdir) / "dataset", "train")
            self.assertEqual(index, [])

    def test_different_overlay_with_motion_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "train" / "00001" / "0001"
            (root / "overlays_masks").mkdir(parents=True)
            write_rgb(root / "I0.png", 100)
            write_rgb(root / "I1.png", 120)
            write_png(root / "overlays_masks" / "000_I0.png", 255)
            write_png(root / "overlays_masks" / "000_I1.png", 255)
            metadata = {
                "split": "train",
                "source_rel": "00001/0001",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "000",
                        "center0": [16.0, 16.0],
                        "center1": [18.0, 16.0],
                        "geometry": {"mode": "motion", "motion_step": [0.1, 0.0]},
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    },
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))

            self.assertTrue(_overlay_has_motion(metadata["overlays"][0]))
            index = build_pair_index(Path(tmpdir) / "dataset", "train")
            self.assertEqual(index, [])

    def test_overlay_with_edge_adjacent_center_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "train" / "00001" / "0001"
            (root / "overlays_masks").mkdir(parents=True)
            write_rgb(root / "I0.png", 100)
            write_rgb(root / "I1.png", 120)
            write_png(root / "overlays_masks" / "000_I0.png", 255)
            write_png(root / "overlays_masks" / "000_I1.png", 255)
            metadata = {
                "split": "train",
                "source_rel": "00001/0001",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "000",
                        "center0": [1.5, 16.0],
                        "center1": [16.0, 16.0],
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            index = build_pair_index(Path(tmpdir) / "dataset", "train")
            self.assertEqual(index, [])

    def test_geometry_center_outside_frame_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "val" / "00077" / "0105"
            (root / "overlays_masks").mkdir(parents=True)
            Image.new("RGB", (448, 256), color=(100, 100, 100)).save(root / "I0.png")
            Image.new("RGB", (448, 256), color=(120, 120, 120)).save(root / "I1.png")
            write_png(root / "overlays_masks" / "004_I0.png", 255)
            write_png(root / "overlays_masks" / "004_I1.png", 255)
            metadata = {
                "source_rel": "00077/0105",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "004",
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "geometry": {
                            "center0": [416.5, 273.0],
                            "center1": [416.5, 273.0],
                        },
                        "mask_paths": {"I0": "overlays_masks/004_I0.png", "I1": "overlays_masks/004_I1.png"},
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))
            index = build_pair_index(Path(tmpdir) / "dataset", "val")
            self.assertEqual(index, [])

    def test_train_sampling_can_swap_endpoint_order_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset" / "train" / "00001" / "0001"
            (root / "overlays_masks").mkdir(parents=True)
            write_rgb(root / "I0.png", 40)
            write_rgb(root / "I1.png", 220)
            write_png(root / "overlays_masks" / "000_I0.png", 255)
            write_png(root / "overlays_masks" / "000_I1.png", 255)
            metadata = {
                "split": "train",
                "source_rel": "00001/0001",
                "scenechange": None,
                "overlays": [
                    {
                        "object_id": "000",
                        "center0": [16.0, 16.0],
                        "center1": [16.0, 16.0],
                        "temporal": {"mode": "change_appearance", "detail": {"variant": "textual"}},
                        "mask_paths": {"I0": "overlays_masks/000_I0.png", "I1": "overlays_masks/000_I1.png"},
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata))

            dataset = VimeoOverlayPairDataset(
                Path(tmpdir) / "dataset",
                "train",
                image_size=28,
                swap_pair_probability=1.0,
            )
            sample = dataset[0]
            self.assertGreater(sample["image1"].mean().item(), sample["image2"].mean().item())

    def test_distinct_train_subset_shortfall_uses_all_valid_indices(self) -> None:
        config = {
            "data": {
                "max_train_samples": 4,
                "subset_seed": 0,
                "distinct_train_subset_per_epoch": True,
            }
        }
        dataset = DummyDataset(split="train", valid_count=10, total_count=12)
        epoch_subsets = prepare_epoch_subsets(config, dataset, "train", 3)
        self.assertIsNotNone(epoch_subsets)
        flattened = [index for subset in epoch_subsets for index in subset]
        self.assertEqual(len(flattened), 10)
        self.assertEqual(len(set(flattened)), 10)
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual([len(subset) for subset in epoch_subsets], [4, 3, 3])

    def test_full_dataset_run_prefilters_to_valid_indices(self) -> None:
        config = {"data": {"subset_seed": 0}}
        dataset = DummyDataset(split="train", valid_count=3, total_count=5)
        epoch_subsets = prepare_epoch_subsets(config, dataset, "train", 2)
        self.assertIsNotNone(epoch_subsets)
        self.assertEqual(epoch_subsets, [[0, 1, 2], [0, 1, 2]])

    def test_pos_weight_auto_can_cap_valid_overlay_count(self) -> None:
        config = {"train": {"pos_weight": "auto", "pos_weight_max_valid_samples": 3}}
        dataset = DummyLabeledDataset(labels=[1, 0, 0, 1, 1], valid_indices={0, 1, 2, 3, 4})
        value = resolve_pos_weight_value(config, train_dataset=dataset)
        self.assertEqual(value, 2.0)
        self.assertEqual(config["train"]["resolved_pos_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
