#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO


OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Test" / "GOSeg_Inconsistencies"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

sys.path.insert(0, str(REPO_ROOT / "GOSeg"))
from tracked_inference import load_efficient_sam_model, process_tracked_images, resolve_image_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tracked GOSeg over Inconsistencies outputs and export a hackview under Test/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=OUTPUTS_ROOT, help="Root of Inconsistencies triplet outputs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root under Test/ for GOSeg outputs")
    parser.add_argument("--yolo-model", type=Path, default=REPO_ROOT / "GOSeg" / "best_60.pt", help="YOLO weights")
    parser.add_argument("--sam-type", type=str, default="vits", choices=["vitt", "vits"], help="EfficientSAM model type")
    parser.add_argument("--sam-model", type=Path, default=REPO_ROOT / "GOSeg" / "efficient_sam_vits.pt", help="EfficientSAM weights")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Ultralytics tracker config")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="Class IDs to track")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="Tracking IoU threshold")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[256, 448], help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="", help="Inference device, e.g. cpu, mps, 0")
    parser.add_argument("--half", action="store_true", help="Use FP16 in YOLO tracking")
    parser.add_argument("--agnostic-nms", action="store_true", help="Enable class-agnostic NMS")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay transparency")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--max-sequences", type=int, default=0, help="Optional cap for smoke runs")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse output root if it exists")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg:
        if device_arg.isdigit():
            return torch.device(f"cuda:{device_arg}")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_clean_dir(path: Path, exist_ok: bool) -> None:
    if path.exists():
        if not exist_ok:
            shutil.rmtree(path)
        elif not path.is_dir():
            raise NotADirectoryError(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_sequence_dirs(source_root: Path) -> list[Path]:
    sequence_dirs: list[Path] = []
    for dataset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for seq_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
                sequence_dirs.append(seq_dir)
    return sequence_dirs


def copy_hackview_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_hackview_source(run_dir: Path, frame_stem: str) -> Path:
    frame_dir = run_dir / "frames" / frame_stem
    candidates = [
        frame_dir / f"{frame_stem}_combined.jpg",
        frame_dir / f"{frame_stem}_detection.jpg",
        frame_dir / f"{frame_stem}_original.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing hackview frame for {frame_stem} in {run_dir}")


def export_hackview(output_root: Path, runs_root: Path, sequence_dirs: list[Path]) -> None:
    hackview_root = output_root / "hackview"
    ensure_clean_dir(hackview_root, exist_ok=False)

    grouped: dict[str, list[Path]] = {}
    for seq_dir in sequence_dirs:
        dataset_name = seq_dir.parent.parent.name
        grouped.setdefault(dataset_name, []).append(seq_dir)

    for dataset_name, seqs in grouped.items():
        dataset_out = hackview_root / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)
        for seq_idx, seq_dir in enumerate(seqs):
            video_name = seq_dir.parent.name
            seq_name = seq_dir.name
            run_dir = runs_root / dataset_name / video_name / seq_name
            for frame_idx, frame_stem in enumerate(("I1", "I2")):
                src = find_hackview_source(run_dir, frame_stem)
                dst_name = f"{seq_idx:02d}_{frame_idx}_{video_name}_{seq_name}{src.suffix.lower()}"
                copy_hackview_frame(src, dataset_out / dst_name)


def make_run_args(args: argparse.Namespace, seq_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=str(seq_dir),
        tracker=args.tracker,
        classes=args.classes,
        imgsz=args.imgsz,
        iou=args.iou,
        conf=args.conf,
        agnostic_nms=args.agnostic_nms,
        half=args.half,
        device=args.device,
        alpha=args.alpha,
        line_width=args.line_width,
        save_overlay=True,
    )


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    runs_root = output_root / "runs"

    ensure_clean_dir(output_root, exist_ok=args.exist_ok)
    ensure_clean_dir(runs_root, exist_ok=True)

    sequence_dirs = iter_sequence_dirs(source_root)
    if args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]
    if not sequence_dirs:
        raise ValueError(f"No sequence directories found under {source_root}")

    print(f"Loading YOLO weights from {args.yolo_model}")
    yolo_model = YOLO(str(args.yolo_model))
    print(f"Loading EfficientSAM weights from {args.sam_model}")
    sam_model = load_efficient_sam_model(args.sam_type, str(args.sam_model), device)

    manifest: list[dict[str, str | int]] = []

    for index, seq_dir in enumerate(sequence_dirs, start=1):
        dataset_name = seq_dir.parent.parent.name
        video_name = seq_dir.parent.name
        seq_name = seq_dir.name
        run_dir = runs_root / dataset_name / video_name / seq_name
        run_dir.mkdir(parents=True, exist_ok=True)
        image_paths = resolve_image_paths(seq_dir)
        run_args = make_run_args(args, seq_dir)
        print(f"[{index}/{len(sequence_dirs)}] {dataset_name}/{video_name}/{seq_name}")
        process_tracked_images(image_paths, yolo_model, sam_model, run_args, device, run_dir)
        manifest.append(
            {
                "dataset": dataset_name,
                "video": video_name,
                "sequence": seq_name,
                "run_dir": str(run_dir),
                "frame_count": len(image_paths),
            }
        )

    export_hackview(output_root, runs_root, sequence_dirs)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved runs to {runs_root}")
    print(f"Saved hackview to {output_root / 'hackview'}")


if __name__ == "__main__":
    main()
