from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask(path: Path, threshold: float = 0.0) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    mask = (array > threshold).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).contiguous()


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius)
    return (pooled > 0).to(mask.dtype).squeeze(0)


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    inv = 1.0 - mask
    pooled = F.max_pool2d(inv.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius)
    return (1.0 - (pooled > 0).to(mask.dtype)).squeeze(0)


def build_transition_band(instance_masks: list[torch.Tensor], r1: int, r2: int) -> torch.Tensor:
    band = torch.zeros_like(instance_masks[0])
    for mask in instance_masks:
        outer = dilate(mask, r2)
        inner = erode(mask, r1)
        band = torch.maximum(band, (outer - inner).clamp(0.0, 1.0))
    return band


@dataclass(frozen=True)
class MaskConfig:
    editable_radius: int = 8
    transition_erode_radius: int = 1
    transition_dilate_radius: int = 6


class VimeoOverlayRefineDataset(Dataset):
    """
    Dataset for `Datasets/vimeo_aug_interpolated_for_refinement_training`.

    Exposed inputs:
    - I0.png
    - I_0.5_inter_copied.png
    - M.png

    Training target:
    - I_0.5_copied.png

    Derived masks:
    - E = dilate(M, r_e)
    - T from selected per-overlay masks listed in `_manifest.json`
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mask_config: MaskConfig | None = None,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.split_root = self.root / split
        self.mask_config = mask_config or MaskConfig()

        if not self.split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_root}")

        self.items = self._load_manifest_items()
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise RuntimeError(f"No valid samples found under {self.split_root}")

    def _load_manifest_items(self) -> list[dict]:
        manifest_path = self.split_root / "_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        valid_items = []
        for item in manifest.get("items", []):
            rel = item["rel"]
            sample_dir = self.split_root / rel
            required = [
                sample_dir / "I0.png",
                sample_dir / "I_0.5_inter_copied.png",
                sample_dir / "I_0.5_copied.png",
                sample_dir / "M.png",
            ]
            if not all(path.exists() for path in required):
                continue
            valid_items.append(item)
        return valid_items

    def __len__(self) -> int:
        return len(self.items)

    def _load_transition_instance_masks(self, sample_dir: Path, overlay_ids: list[str]) -> list[torch.Tensor]:
        peroverlay_dir = sample_dir / "peroverlay"
        instance_masks = []
        for object_id in overlay_ids:
            mask_i0 = peroverlay_dir / f"{object_id}_I0.png"
            mask_i1 = peroverlay_dir / f"{object_id}_I1.png"
            if not mask_i0.exists() or not mask_i1.exists():
                continue
            instance_masks.append(torch.maximum(load_mask(mask_i0), load_mask(mask_i1)))
        return instance_masks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        rel = item["rel"]
        sample_dir = self.split_root / rel

        i0 = load_rgb(sample_dir / "I0.png")
        i05_cp = load_rgb(sample_dir / "I_0.5_inter_copied.png")
        i05_gt = load_rgb(sample_dir / "I_0.5_copied.png")
        m = load_mask(sample_dir / "M.png")
        e = dilate(m, self.mask_config.editable_radius)

        overlay_ids = [str(object_id) for object_id in item.get("selected_object_ids", [])]
        instance_masks = self._load_transition_instance_masks(sample_dir, overlay_ids)
        t = build_transition_band(
            instance_masks,
            r1=self.mask_config.transition_erode_radius,
            r2=self.mask_config.transition_dilate_radius,
        ) if instance_masks else e.clone()

        return {
            "i0": i0,
            "i05_cp": i05_cp,
            "i05_gt": i05_gt,
            "m": m,
            "e": e,
            "t": t,
            "sample_id": rel,
            "active_overlay_ids": ",".join(overlay_ids),
        }
