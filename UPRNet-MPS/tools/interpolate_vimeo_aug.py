#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from core.device import resolve_device
from core.pipeline import Pipeline


def list_sequence_dirs(root: Path, limit: int | None) -> list[Path]:
    seqs = sorted(path for path in root.glob("*/*") if path.is_dir())
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def infer_middle_frame(ppl: Pipeline, frame0, frame1, device: torch.device):
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


def main():
    parser = argparse.ArgumentParser(description="Batch interpolate Vimeo augmented triplets with UPR-Net.")
    parser.add_argument("--input-root", type=Path, required=True, help="Input split root, e.g. Datasets/vimeo_triplet_augmented_full/train")
    parser.add_argument("--output-root", type=Path, required=True, help="Output split root, e.g. Datasets/vimeo_aug_interpolated/train")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of sequences to process")
    parser.add_argument("--device", type=str, default="auto", help="runtime device: auto, mps, cpu, cuda, or cuda:N")
    parser.add_argument("--model_size", type=str, default="base", help="One of: base, large, LARGE")
    parser.add_argument("--model_file", type=str, default="./checkpoints/upr-base.pkl", help="Path to model weights")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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

    seq_dirs = list_sequence_dirs(input_root, args.limit)
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
        interp_path = out_dir / "I_0.5_upr.png"
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
        with flow_path.open("wb") as handle:
            import numpy as np
            np.save(handle, bi_flow)

        manifest["items"].append({"rel": rel.as_posix(), "status": "ok"})
        if idx % 10 == 0 or idx == len(seq_dirs):
            print(f"Processed {idx}/{len(seq_dirs)}: {rel}")

    manifest_path = output_root / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote interpolated outputs to {output_root}")


if __name__ == "__main__":
    main()

