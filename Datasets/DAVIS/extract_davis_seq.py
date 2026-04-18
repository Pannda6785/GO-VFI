#!/usr/bin/env python3
"""Create non-overlapping DAVIS clips from JPEGImages/480p."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect_frames(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def clip_dataset_name(length: int) -> str:
    if length == 3:
        return "triplets"
    if length == 7:
        return "septuplets"
    return f"{length}-frame clips"


def default_dest_name(length: int) -> str:
    if length == 3:
        return "DAVIS-Triplet"
    if length == 7:
        return "DAVIS-Septuplet"
    return f"DAVIS-{length}-Frame"


def build_clips(source_root: Path, dest_root: Path, clip_length: int) -> tuple[int, int]:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")

    dest_root.mkdir(parents=True, exist_ok=True)

    video_count = 0
    clip_count = 0

    for video_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        frames = collect_frames(video_dir)
        if len(frames) < clip_length:
            continue

        video_count += 1
        video_dest = dest_root / video_dir.name
        video_dest.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(frames) - clip_length + 1, clip_length):
            clip_frames = frames[start : start + clip_length]
            clip_name = f"{clip_frames[0].stem}_{clip_frames[-1].stem}"
            clip_dir = video_dest / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)

            for idx, frame_path in enumerate(clip_frames):
                target = clip_dir / f"{idx:05d}{frame_path.suffix.lower()}"
                shutil.copy2(frame_path, target)

            clip_count += 1

    return video_count, clip_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DAVIS JPEGImages/480p videos into non-overlapping clips."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("DAVIS/JPEGImages/480p"),
        help="Path to DAVIS JPEGImages/480p.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=7,
        help="Number of frames per clip. Common values are 3 and 7.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output directory for the clip dataset. Defaults to a length-specific DAVIS-* directory.",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    dest_arg = args.dest if args.dest is not None else Path(default_dest_name(args.clip_length))
    dest_root = dest_arg.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    videos, clips = build_clips(source_root, dest_root, args.clip_length)
    print(f"Processed {videos} videos into {clips} {clip_dataset_name(args.clip_length)} at {dest_root}")


if __name__ == "__main__":
    main()
