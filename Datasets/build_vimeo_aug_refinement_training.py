#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_size_arg(size_arg: str) -> int | None:
    if size_arg == "full":
        return None
    size = int(size_arg)
    if size <= 0:
        raise ValueError("--size must be a positive integer or 'full'")
    return size


def list_sequence_dirs(root: Path, limit: int | None) -> list[Path]:
    seqs = sorted(path for path in root.glob("*/*") if path.is_dir())
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {path}")
    return mask


def union_discontinuous_masks(src_dir: Path, metadata: dict) -> tuple[np.ndarray, list[dict]]:
    selected_overlays: list[dict] = []
    mask_union: np.ndarray | None = None

    for overlay in metadata.get("overlays", []):
        mode = overlay.get("mode") or overlay.get("temporal", {}).get("mode")
        if mode not in {"appear_disappear", "change_appearance"}:
            continue
        selected_overlays.append(overlay)
        for rel_path in overlay.get("mask_paths", {}).values():
            mask = read_mask(src_dir / rel_path)
            if mask_union is None:
                mask_union = np.zeros_like(mask, dtype=np.uint8)
            mask_union = np.maximum(mask_union, mask)

    if mask_union is None:
        aggregate = read_mask(src_dir / "aggregate_masks" / "M05_copied.png")
        mask_union = np.zeros_like(aggregate, dtype=np.uint8)

    return mask_union, selected_overlays


def build_inter_copied(i0: np.ndarray, inter: np.ndarray, mask: np.ndarray) -> np.ndarray:
    select = mask > 0
    out = inter.copy()
    out[select] = i0[select]
    return out


def copy_folder(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build refinement-training samples from Vimeo augmented and UPR outputs.")
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "Datasets" / "vimeo_triplet_augmented_full" / "train")
    parser.add_argument("--inter-root", type=Path, default=REPO_ROOT / "Datasets" / "vimeo_aug_interpolated" / "train")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "Datasets" / "vimeo_aug_interpolated_for_refinement_training" / "train")
    parser.add_argument("--size", type=str, default="300", help="Number of sequences to process, or 'full'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing packaged samples.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    inter_root = args.inter_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    limit = parse_size_arg(args.size)
    seq_dirs = list_sequence_dirs(source_root, limit)

    manifest = {
        "source_root": str(source_root),
        "inter_root": str(inter_root),
        "output_root": str(output_root),
        "count": len(seq_dirs),
        "items": [],
    }

    for idx, src_dir in enumerate(seq_dirs, start=1):
        rel = src_dir.relative_to(source_root)
        inter_dir = inter_root / rel
        out_dir = output_root / rel
        metadata_path = src_dir / "metadata.json"
        inter_candidates = [
            inter_dir / "I_0.5_inter.png",
            inter_dir / "I_0.5_upr.png",
        ]
        inter_path = next((path for path in inter_candidates if path.exists()), inter_candidates[0])

        if not metadata_path.exists() or not inter_path.exists():
            manifest["items"].append({"rel": rel.as_posix(), "status": "missing_input"})
            continue

        if out_dir.exists() and not args.overwrite:
            manifest["items"].append({"rel": rel.as_posix(), "status": "skipped_existing"})
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = json.loads(metadata_path.read_text())
        i0 = cv2.imread(str(src_dir / "I0.png"), cv2.IMREAD_COLOR)
        copied = cv2.imread(str(src_dir / "I_0.5_copied.png"), cv2.IMREAD_COLOR)
        inter = cv2.imread(str(inter_path), cv2.IMREAD_COLOR)
        if i0 is None or copied is None or inter is None:
            manifest["items"].append({"rel": rel.as_posix(), "status": "missing_image"})
            continue

        mask_union, selected_overlays = union_discontinuous_masks(src_dir, metadata)
        inter_copied = build_inter_copied(i0, inter, mask_union)

        cv2.imwrite(str(out_dir / "I0.png"), i0)
        cv2.imwrite(str(out_dir / "I_0.5_copied.png"), copied)
        cv2.imwrite(str(out_dir / "I_0.5_inter.png"), inter)
        cv2.imwrite(str(out_dir / "I_0.5_inter_copied.png"), inter_copied)
        cv2.imwrite(str(out_dir / "M.png"), mask_union)

        copy_folder(src_dir / "aggregate_masks", out_dir / "aggregate")
        copy_folder(src_dir / "overlays_masks", out_dir / "peroverlay")

        refinement_meta = {
            "source_rel": metadata.get("source_rel"),
            "split": metadata.get("split"),
            "scenechange": metadata.get("scenechange"),
            "composite_profile": metadata.get("composite_profile"),
            "tasks": metadata.get("tasks"),
            "selected_overlay_modes": ["appear_disappear", "change_appearance"],
            "selected_object_ids": [overlay.get("object_id") for overlay in selected_overlays],
            "selected_overlays": selected_overlays,
            "files": {
                "I0": "I0.png",
                "I_0.5_copied": "I_0.5_copied.png",
                "I_0.5_inter": "I_0.5_inter.png",
                "I_0.5_inter_copied": "I_0.5_inter_copied.png",
                "M": "M.png",
                "aggregate": "aggregate",
                "peroverlay": "peroverlay",
            },
            "mask_build_rule": {
                "description": "Union of all per-overlay masks for overlays whose mode is appear_disappear or change_appearance.",
                "source_mask_frames": ["I0", "I_0.5", "I1", "I_0.5_copied"],
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(refinement_meta, indent=2), encoding="utf-8")
        manifest["items"].append({"rel": rel.as_posix(), "status": "ok", "selected_object_ids": refinement_meta["selected_object_ids"]})

        if idx % 10 == 0 or idx == len(seq_dirs):
            print(f"Packaged {idx}/{len(seq_dirs)}: {rel}")

    (output_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote refinement dataset to {output_root}")


if __name__ == "__main__":
    main()
