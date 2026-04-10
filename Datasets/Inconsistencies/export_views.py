#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
CLEANED_ROOT = ROOT / "cleaned"
HACKVIEW_ROOT = ROOT / "hackview"
FRAME_NAMES = ["I0", "I1", "I2"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_frame_path(seq_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = seq_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem} frame in {seq_dir}")


def dataset_dirs() -> list[Path]:
    return sorted(path for path in OUTPUTS_ROOT.iterdir() if path.is_dir())


def sequence_dirs(dataset_dir: Path) -> list[Path]:
    items: list[Path] = []
    for video_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for seq_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            items.append(seq_dir)
    return items


def export_cleaned() -> None:
    ensure_empty_dir(CLEANED_ROOT)
    for dataset_dir in dataset_dirs():
        for seq_dir in sequence_dirs(dataset_dir):
            rel = seq_dir.relative_to(OUTPUTS_ROOT)
            out_dir = CLEANED_ROOT / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            for stem in FRAME_NAMES:
                src = find_frame_path(seq_dir, stem)
                shutil.copy2(src, out_dir / src.name)


def export_hackview() -> None:
    ensure_empty_dir(HACKVIEW_ROOT)
    for dataset_dir in dataset_dirs():
        out_dir = HACKVIEW_ROOT / dataset_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for seq_idx, seq_dir in enumerate(sequence_dirs(dataset_dir)):
            video_name = seq_dir.parent.name
            seq_name = seq_dir.name
            for frame_idx, stem in enumerate(FRAME_NAMES):
                src = find_frame_path(seq_dir, stem)
                dst_name = f"{seq_idx:02d}_{frame_idx:01d}_{video_name}_{seq_name}{src.suffix.lower()}"
                shutil.copy2(src, out_dir / dst_name)


def main() -> None:
    export_cleaned()
    export_hackview()
    print(f"Exported cleaned view to {CLEANED_ROOT}")
    print(f"Exported hackview to {HACKVIEW_ROOT}")


if __name__ == "__main__":
    main()
