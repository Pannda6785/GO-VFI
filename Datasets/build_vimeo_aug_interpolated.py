#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
UPR_ROOT = REPO_ROOT / "UPRNet-MPS"
sys.path.insert(0, str(UPR_ROOT))

from core.device import resolve_device  # noqa: E402
from core.pipeline import Pipeline  # noqa: E402


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


def infer_middle_frame(ppl: Pipeline, frame0: np.ndarray, frame1: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    img0 = (torch.tensor(frame0.transpose(2, 0, 1), device=device) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(frame1.transpose(2, 0, 1), device=device) / 255.0).unsqueeze(0)

    _, _, h, w = img0.shape
    pyr_level = math.ceil(math.log2(w / 448) + 3)
    divisor = 2 ** (pyr_level - 1 + 2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    interp_img, bi_flow = ppl.inference(img0, img1, time_period=0.5, pyr_level=pyr_level)
    interp_img = interp_img[:, :, :h, :w]
    bi_flow = bi_flow[:, :, :h, :w]

    out_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    out_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)
    return out_img, out_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UPR interpolations for the Vimeo augmented dataset.")
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "Datasets" / "vimeo_triplet_augmented_full" / "train")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "Datasets" / "vimeo_aug_interpolated" / "train")
    parser.add_argument("--size", type=str, default="300", help="Number of sequences to process, or 'full'.")
    parser.add_argument("--device", type=str, default="auto", help="runtime device: auto, mps, cpu, cuda, or cuda:N")
    parser.add_argument("--model-size", type=str, default="base", help="One of: base, large, LARGE")
    parser.add_argument("--model-file", type=str, default=str(UPR_ROOT / "checkpoints" / "upr-base.pkl"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    limit = parse_size_arg(args.size)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_cfg_dict = {
        "load_pretrain": True,
        "model_size": args.model_size,
        "model_file": args.model_file,
        "device": args.device,
    }
    torch.set_grad_enabled(False)
    ppl = Pipeline(model_cfg_dict)
    ppl.eval()

    seq_dirs = list_sequence_dirs(input_root, limit)
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "device": str(device),
        "model_size": args.model_size,
        "model_file": args.model_file,
        "count": len(seq_dirs),
        "items": [],
    }

    for idx, seq_dir in enumerate(seq_dirs, start=1):
        rel = seq_dir.relative_to(input_root)
        out_dir = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        interp_path = out_dir / "I_0.5_inter.png"
        flow_path = out_dir / "bi_flow.npy"
        if interp_path.exists() and flow_path.exists() and not args.overwrite:
            manifest["items"].append({"rel": rel.as_posix(), "status": "skipped_existing"})
            continue

        frame0 = cv2.imread(str(seq_dir / "I0.png"), cv2.IMREAD_COLOR)
        frame1 = cv2.imread(str(seq_dir / "I1.png"), cv2.IMREAD_COLOR)
        if frame0 is None or frame1 is None:
            manifest["items"].append({"rel": rel.as_posix(), "status": "missing_input"})
            continue

        interp_img, bi_flow = infer_middle_frame(ppl, frame0, frame1, device)
        cv2.imwrite(str(interp_path), interp_img)
        np.save(flow_path, bi_flow)
        manifest["items"].append({"rel": rel.as_posix(), "status": "ok"})

        if idx % 10 == 0 or idx == len(seq_dirs):
            print(f"Processed {idx}/{len(seq_dirs)}: {rel}")

    (output_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote interpolated outputs to {output_root}")


if __name__ == "__main__":
    main()

