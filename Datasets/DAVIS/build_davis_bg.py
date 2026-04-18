#!/usr/bin/env python3
"""Build MAT-based background images for DAVIS triplet items and interpolate the middle background with UPR-Net."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

import config as cfg


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DATASET_ROOT = Path(__file__).resolve().parent
MAT_ROOT = DATASET_ROOT.parent / "MAT"
UPR_ROOT = DATASET_ROOT.parent / "UPR-Net"
DEFAULT_MAT_NETWORK = MAT_ROOT / "pretrained" / "MAT_Places_512_fp16.safetensors"
DEFAULT_UPR_MODEL = UPR_ROOT / "checkpoints" / "upr-base.pkl"


def find_single_frame(seq_dir: Path, stem: str) -> Path:
    matches = [path for path in seq_dir.iterdir() if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTS]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one frame named {stem}.* in {seq_dir}, found {len(matches)}")
    return matches[0]


def find_sequence_dirs(source_root: Path) -> list[Path]:
    sequence_dirs: list[Path] = []
    for aggregate_dir in sorted(source_root.rglob("aggregate_masks")):
        seq_dir = aggregate_dir.parent
        try:
            find_single_frame(seq_dir, "I0")
            find_single_frame(seq_dir, "I1")
            find_single_frame(seq_dir, "I2")
        except FileNotFoundError:
            continue
        for idx in range(3):
            mask_path = aggregate_dir / f"M{idx}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing aggregate mask: {mask_path}")
        sequence_dirs.append(seq_dir)
    return sequence_dirs


def whiten_and_dilate_mask(mask_path: Path, out_path: Path, dilate_pixels: int) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    # Collapse all instance IDs into a single binary inpaint region before dilation.
    binary = (mask > 0).astype("uint8") * 255
    if dilate_pixels > 0:
        kernel_size = dilate_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    if not cv2.imwrite(str(out_path), binary):
        raise RuntimeError(f"Failed to write mask: {out_path}")


def run_mat_on_sequence(
    seq_dir: Path,
    mat_network: Path,
    resolution: int,
    noise_mode: str,
    dilate_pixels: int,
) -> list[Path]:
    aggregate_root = seq_dir / "aggregate_masks"
    frame_paths = [find_single_frame(seq_dir, f"I{idx}") for idx in range(3)]

    with tempfile.TemporaryDirectory(prefix="davis_bg_mat_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        image_root = tmp_dir / "images"
        mask_root = tmp_dir / "masks"
        out_root = tmp_dir / "out"
        image_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)

        for idx, frame_path in enumerate(frame_paths):
            image_copy = image_root / frame_path.name
            shutil.copy2(frame_path, image_copy)
            whiten_and_dilate_mask(aggregate_root / f"M{idx}.png", mask_root / f"I{idx}.png", dilate_pixels)

        cmd = [
            sys.executable,
            "generate_image.py",
            "--network",
            str(mat_network),
            "--dpath",
            str(image_root),
            "--mpath",
            str(mask_root),
            "--mask-dilate",
            "0",
            "--resolution",
            str(resolution),
            "--noise-mode",
            noise_mode,
            "--outdir",
            str(out_root),
        ]
        subprocess.run(cmd, cwd=MAT_ROOT, check=True)

        bg_paths: list[Path] = []
        for idx in range(3):
            tmp_output = out_root / f"I{idx}.png"
            if not tmp_output.exists():
                raise FileNotFoundError(f"MAT output missing: {tmp_output}")
            final_path = seq_dir / f"I{idx}_bg.png"
            shutil.copy2(tmp_output, final_path)
            bg_paths.append(final_path)

    return bg_paths


def run_upr_interpolation(
    seq_dir: Path,
    upr_model_file: Path,
    upr_model_size: str,
) -> Path:
    frame0 = seq_dir / "I0_bg.png"
    frame2 = seq_dir / "I2_bg.png"
    if not frame0.exists() or not frame2.exists():
        raise FileNotFoundError(f"Missing background endpoints in {seq_dir}")

    with tempfile.TemporaryDirectory(prefix="davis_bg_upr_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        env = dict(**__import__("os").environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(UPR_ROOT) if not existing_pythonpath else f"{UPR_ROOT}:{existing_pythonpath}"
        cmd = [
            sys.executable,
            "demo/interp_imgs.py",
            "--frame0",
            str(frame0),
            "--frame1",
            str(frame2),
            "--save_dir",
            str(tmp_dir),
            "--model_size",
            upr_model_size,
            "--model_file",
            str(upr_model_file),
        ]
        subprocess.run(cmd, cwd=UPR_ROOT, env=env, check=True)
        interp_path = tmp_dir / "3-interp-img.png"
        if not interp_path.exists():
            raise FileNotFoundError(f"UPR-Net output missing: {interp_path}")
        final_path = seq_dir / "I1_bg_inter.png"
        shutil.copy2(interp_path, final_path)
        return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAT inpainting on each DAVIS triplet item and interpolate the middle background with UPR-Net."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=cfg.OUTPUT_ROOT,
        help="Root directory containing triplet sequence items with I0/I1/I2 and aggregate_masks.",
    )
    parser.add_argument(
        "--mat-network",
        type=Path,
        default=DEFAULT_MAT_NETWORK,
        help="MAT checkpoint path.",
    )
    parser.add_argument(
        "--upr-model-file",
        type=Path,
        default=DEFAULT_UPR_MODEL,
        help="UPR-Net checkpoint path.",
    )
    parser.add_argument(
        "--upr-model-size",
        default="base",
        choices=["base", "large", "LARGE"],
        help="UPR-Net model size.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="MAT inference resolution.",
    )
    parser.add_argument(
        "--noise-mode",
        default="const",
        choices=["const", "random", "none"],
        help="MAT noise mode.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=12,
        help="Dilate the whitened aggregate mask by this many pixels before inpainting.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    mat_network = args.mat_network.resolve()
    upr_model_file = args.upr_model_file.resolve()

    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if not mat_network.exists():
        raise SystemExit(f"MAT checkpoint does not exist: {mat_network}")
    if not upr_model_file.exists():
        raise SystemExit(
            "UPR-Net checkpoint does not exist: "
            f"{upr_model_file}. Pass --upr-model-file to a valid checkpoint."
        )

    sequence_dirs = find_sequence_dirs(source_root)
    if not sequence_dirs:
        raise SystemExit(f"No sequence directories with aggregate_masks found under {source_root}")

    for seq_dir in sequence_dirs:
        print(f"Processing {seq_dir}")
        run_mat_on_sequence(
            seq_dir=seq_dir,
            mat_network=mat_network,
            resolution=args.resolution,
            noise_mode=args.noise_mode,
            dilate_pixels=args.mask_dilate,
        )
        run_upr_interpolation(
            seq_dir=seq_dir,
            upr_model_file=upr_model_file,
            upr_model_size=args.upr_model_size,
        )

    print(f"Processed {len(sequence_dirs)} sequence(s) under {source_root}")


if __name__ == "__main__":
    main()
