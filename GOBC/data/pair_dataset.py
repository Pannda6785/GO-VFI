from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from GOBC.models.dinov2_backbone import DINO_MEAN, DINO_STD


VALID_SPLITS = {"train", "val", "test"}
CENTER_EDGE_MARGIN_FRACTION = 1.0 / 16.0


@dataclass(frozen=True)
class OverlaySample:
    split: str
    source_rel: str
    object_id: str
    label: int
    temporal_mode: str
    temporal_variant: str | None
    image1_path: str
    image2_path: str
    mask1_path: str
    mask2_path: str


def _load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _has_scene_change(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null", "no"}
    return True


def _temporal_fields(overlay: dict[str, Any]) -> tuple[str | None, str | None]:
    temporal = overlay.get("temporal") or {}
    mode = temporal.get("mode", overlay.get("mode"))
    detail = temporal.get("detail") or {}
    variant = detail.get("variant")
    return mode, variant


def derive_overlay_label(overlay: dict[str, Any]) -> int | None:
    mode, variant = _temporal_fields(overlay)
    detail = ((overlay.get("temporal") or {}).get("detail") or {})
    if mode == "appear_disappear":
        return None
    if mode == "change_appearance":
        if variant == "textual":
            return 1
        if variant == "composite":
            components = detail.get("components") or []
            if "textual" in components:
                return 1
    return 0


def _mask_paths_for_overlay(meta_dir: Path, overlay: dict[str, Any]) -> tuple[Path, Path]:
    mask_paths = overlay.get("mask_paths")
    if mask_paths:
        return meta_dir / mask_paths["I0"], meta_dir / mask_paths["I1"]
    object_id = overlay["object_id"]
    return meta_dir / "overlays_masks" / f"{object_id}_I0.png", meta_dir / "overlays_masks" / f"{object_id}_I1.png"


def _overlay_center(overlay: dict[str, Any], key: str) -> Any:
    if key in overlay:
        return overlay.get(key)
    geometry = overlay.get("geometry") or {}
    return geometry.get(key)


def _center_inside_frame(center: Any, image_size: tuple[int, int]) -> bool:
    if center is None:
        return True
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return False
    x, y = center
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    width, height = image_size
    edge_margin = float(width) * CENTER_EDGE_MARGIN_FRACTION
    x = float(x)
    y = float(y)
    return edge_margin <= x < float(width) - edge_margin and edge_margin <= y < float(height) - edge_margin


def _overlay_centers_inside_frames(
    overlay: dict[str, Any],
    image1_size: tuple[int, int],
    image2_size: tuple[int, int],
) -> bool:
    return _center_inside_frame(_overlay_center(overlay, "center0"), image1_size) and _center_inside_frame(
        _overlay_center(overlay, "center1"), image2_size
    )


def _overlay_has_motion(overlay: dict[str, Any]) -> bool:
    geometry = overlay.get("geometry") or {}
    if str(geometry.get("mode", "")).lower() == "motion":
        return True
    motion_step = geometry.get("motion_step") or ()
    if isinstance(motion_step, (list, tuple)):
        for value in motion_step:
            if isinstance(value, (int, float)) and float(value) != 0.0:
                return True
    return False


def build_pair_index(dataset_root: str | Path, split: str) -> list[OverlaySample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split {split!r}. Expected one of {sorted(VALID_SPLITS)}.")

    root = Path(dataset_root)
    split_root = root / split
    samples: list[OverlaySample] = []

    for metadata_path in sorted(split_root.glob("*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if _has_scene_change(metadata.get("scenechange")):
            continue

        overlays = metadata.get("overlays") or []
        if not overlays:
            continue

        meta_dir = metadata_path.parent
        image1_path = meta_dir / "I0.png"
        image2_path = meta_dir / "I1.png"
        if not image1_path.exists() or not image2_path.exists():
            continue
        image1_size = _load_image_size(image1_path)
        image2_size = _load_image_size(image2_path)
        for overlay in overlays:
            label = derive_overlay_label(overlay)
            if label is None:
                continue
            if label == 1 and _overlay_has_motion(overlay):
                continue

            if not _overlay_centers_inside_frames(overlay, image1_size, image2_size):
                continue

            mask1_path, mask2_path = _mask_paths_for_overlay(meta_dir, overlay)
            if not all(path.exists() for path in (image1_path, image2_path, mask1_path, mask2_path)):
                continue

            temporal_mode, temporal_variant = _temporal_fields(overlay)
            samples.append(
                OverlaySample(
                    split=split,
                    source_rel=metadata.get("source_rel", f"{metadata_path.parent.parent.name}/{metadata_path.parent.name}"),
                    object_id=str(overlay["object_id"]),
                    label=label,
                    temporal_mode=temporal_mode or "none",
                    temporal_variant=temporal_variant,
                    image1_path=str(image1_path),
                    image2_path=str(image2_path),
                    mask1_path=str(mask1_path),
                    mask2_path=str(mask2_path),
                )
            )

    return samples


def _compute_union_bbox(mask1: np.ndarray, mask2: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    union = np.logical_or(mask1 > 0, mask2 > 0)
    ys, xs = np.nonzero(union)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty union crop: both endpoint masks are empty.")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    height, width = union.shape
    pad_y = max(1, int(round((y1 - y0) * margin)))
    pad_x = max(1, int(round((x1 - x0) * margin)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


class VimeoOverlayPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int = 518,
        crop_margin: float = 0.15,
        normalize: bool = True,
        patch_size: int = 14,
        min_patch_tokens: int = 1,
        mask_threshold: float = 0.3,
        swap_pair_probability: float = 0.5,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.normalize = normalize
        self.patch_size = patch_size
        self.min_patch_tokens = min_patch_tokens
        self.mask_threshold = float(mask_threshold)
        self.swap_pair_probability = float(swap_pair_probability)
        self.samples = build_pair_index(dataset_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_mask(path: str) -> Image.Image:
        return Image.open(path).convert("L")

    def _prepare_tensors(self, sample: OverlaySample, allow_swap: bool = True) -> dict[str, Any]:
        image1 = self._load_rgb(sample.image1_path)
        image2 = self._load_rgb(sample.image2_path)
        mask1 = self._load_mask(sample.mask1_path)
        mask2 = self._load_mask(sample.mask2_path)

        mask1_np = np.array(mask1)
        mask2_np = np.array(mask2)
        crop_box = _compute_union_bbox(mask1_np, mask2_np, self.crop_margin)

        image1 = image1.crop(crop_box).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image2 = image2.crop(crop_box).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask1 = mask1.crop(crop_box).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask2 = mask2.crop(crop_box).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image1_tensor = TF.to_tensor(image1)
        image2_tensor = TF.to_tensor(image2)
        if self.normalize:
            image1_tensor = TF.normalize(image1_tensor, DINO_MEAN, DINO_STD)
            image2_tensor = TF.normalize(image2_tensor, DINO_MEAN, DINO_STD)

        mask1_tensor = (TF.to_tensor(mask1) > 0.5).float()
        mask2_tensor = (TF.to_tensor(mask2) > 0.5).float()

        if (
            allow_swap
            and self.split == "train"
            and self.swap_pair_probability > 0.0
            and torch.rand(()) < self.swap_pair_probability
        ):
            image1_tensor, image2_tensor = image2_tensor, image1_tensor
            mask1_tensor, mask2_tensor = mask2_tensor, mask1_tensor

        self._validate_patch_visibility(mask1_tensor, mask2_tensor)

        return {
            "image1": image1_tensor,
            "image2": image2_tensor,
            "mask1": mask1_tensor,
            "mask2": mask2_tensor,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "source_rel": sample.source_rel,
            "object_id": sample.object_id,
            "temporal_mode": sample.temporal_mode,
            "temporal_variant": sample.temporal_variant or "",
        }

    def _validate_patch_visibility(self, mask1_tensor: torch.Tensor, mask2_tensor: torch.Tensor) -> None:
        grid_hw = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        patch_mask1 = F.interpolate(mask1_tensor.unsqueeze(0), size=grid_hw, mode="area").flatten(2).squeeze(0)
        patch_mask2 = F.interpolate(mask2_tensor.unsqueeze(0), size=grid_hw, mode="area").flatten(2).squeeze(0)
        if int((patch_mask1 > self.mask_threshold).sum().item()) < self.min_patch_tokens:
            raise ValueError("Mask1 collapsed to too few DINO patch tokens after preprocessing.")
        if int((patch_mask2 > self.mask_threshold).sum().item()) < self.min_patch_tokens:
            raise ValueError("Mask2 collapsed to too few DINO patch tokens after preprocessing.")

    def is_valid_index(self, index: int) -> bool:
        try:
            self._prepare_tensors(self.samples[index], allow_swap=False)
        except (OSError, ValueError):
            return False
        return True

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._prepare_tensors(self.samples[index], allow_swap=True)
