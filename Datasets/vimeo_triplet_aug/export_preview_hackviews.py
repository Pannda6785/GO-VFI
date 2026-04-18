#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def valid_sequence_dirs(split_root: Path) -> list[Path]:
    dirs = []
    for path in sorted(split_root.glob("*/*")):
        if not path.is_dir():
            continue
        required = [
            path / "I0.png",
            path / "I_0.5.png",
            path / "I_0.5_copied.png",
            path / "I1.png",
        ]
        if all(p.exists() for p in required):
            dirs.append(path)
    return dirs


def export_hackview(sequence_dirs: list[Path], out_dir: Path, middle_name: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq_idx, seq_dir in enumerate(sequence_dirs):
        rel = seq_dir.relative_to(seq_dir.parents[1])
        stem = f"{seq_idx:02d}_{rel.parts[0]}_{rel.parts[1]}"
        mapping = [
            ("0", seq_dir / "I0.png"),
            ("1", seq_dir / middle_name),
            ("2", seq_dir / "I1.png"),
        ]
        for frame_idx, src in mapping:
            dst = out_dir / f"{stem}_{frame_idx}.png"
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    split_root = args.root / args.split
    seqs = valid_sequence_dirs(split_root)
    legacy_normal = args.root / f"{args.split}_hackview_normal"
    if legacy_normal.exists():
        shutil.rmtree(legacy_normal)
    export_hackview(seqs, args.root / f"{args.split}_hackview_smooth", "I_0.5.png")
    export_hackview(seqs, args.root / f"{args.split}_hackview_copied", "I_0.5_copied.png")
    print(f"Exported {len(seqs)} sequences into hackviews under {args.root}")


if __name__ == "__main__":
    main()
