#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
DOWNLOADS_ROOT = Path("/var/home/anntynn/Downloads")
SOURCES = {
    "hackview_eden": DOWNLOADS_ROOT / "EDEN" / "aug_clean" / "pred",
    "hackview_mambavfi": DOWNLOADS_ROOT / "MambaVFI" / "aug_clean" / "pred",
}
FRAME_STEMS = ["I0", "I2"]
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


def find_prediction_path(pred_root: Path, dataset_name: str, video_name: str, seq_name: str) -> Path | None:
    candidate = pred_root / dataset_name / video_name / f"{seq_name}.png"
    if candidate.exists():
        return candidate
    return None


def dataset_dirs() -> list[Path]:
    return sorted(path for path in OUTPUTS_ROOT.iterdir() if path.is_dir())


def sequence_dirs(dataset_dir: Path) -> list[Path]:
    items: list[Path] = []
    for video_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for seq_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            items.append(seq_dir)
    return items


def export_source(target_name: str, pred_root: Path) -> int:
    out_root = ROOT / target_name
    ensure_empty_dir(out_root)
    fallback_count = 0
    for dataset_dir in dataset_dirs():
        dataset_name = dataset_dir.name
        dataset_out = out_root / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)
        for seq_idx, seq_dir in enumerate(sequence_dirs(dataset_dir)):
            video_name = seq_dir.parent.name
            seq_name = seq_dir.name
            frame0 = find_frame_path(seq_dir, "I0")
            frame1 = find_frame_path(seq_dir, "I1")
            frame2 = find_frame_path(seq_dir, "I2")
            pred = find_prediction_path(pred_root, dataset_name, video_name, seq_name)
            if pred is None:
                pred = frame1
                fallback_count += 1
            files = [frame0, pred, frame2]
            for frame_idx, src in enumerate(files):
                dst_name = f"{seq_idx:02d}_{frame_idx}_{video_name}_{seq_name}{src.suffix.lower()}"
                shutil.copy2(src, dataset_out / dst_name)
    return fallback_count


def main() -> None:
    for target_name, pred_root in SOURCES.items():
        fallback_count = export_source(target_name, pred_root)
        print(f"Exported {target_name} to {ROOT / target_name} with {fallback_count} fallback middle frames")


if __name__ == "__main__":
    main()
