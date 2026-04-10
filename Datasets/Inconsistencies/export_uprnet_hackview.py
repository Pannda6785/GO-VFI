#!/usr/bin/env python3
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
HACKVIEW_UPR_ROOT = ROOT / "hackview_uprnet"
UPR_ROOT = ROOT.parent.parent / "UPR-Net"
UPR_MODEL = UPR_ROOT / "checkpoints" / "upr-base.pkl"
FRAME_NAMES = ["I0", "I1", "I2"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

sys.path.insert(0, str(UPR_ROOT))
from core.pipeline import Pipeline  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path, quality=95)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array(Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)


def init_pipeline() -> Pipeline:
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    width_probe = 854
    del width_probe
    model_cfg_dict = {
        "load_pretrain": True,
        "model_size": "base",
        "model_file": str(UPR_MODEL),
    }
    ppl = Pipeline(model_cfg_dict)
    ppl.eval()
    return ppl


def interpolate_middle_frame(ppl: Pipeline, img0: np.ndarray, img2: np.ndarray) -> np.ndarray:
    tensor0 = (torch.tensor(img0.transpose(2, 0, 1)).to(DEVICE) / 255.0).unsqueeze(0)
    tensor2 = (torch.tensor(img2.transpose(2, 0, 1)).to(DEVICE) / 255.0).unsqueeze(0)
    _, _, h, w = tensor0.shape
    pyr_level = math.ceil(math.log2(w / 448) + 3)
    divisor = 2 ** (pyr_level - 1 + 2)
    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        tensor0 = F.pad(tensor0, padding, "constant", 0.5)
        tensor2 = F.pad(tensor2, padding, "constant", 0.5)
    interp_img, _ = ppl.inference(tensor0, tensor2, time_period=0.5, pyr_level=pyr_level)
    interp_img = interp_img[:, :, :h, :w]
    return (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)


def export_hackview_uprnet() -> None:
    ensure_empty_dir(HACKVIEW_UPR_ROOT)
    ppl = init_pipeline()
    for dataset_dir in dataset_dirs():
        out_dir = HACKVIEW_UPR_ROOT / dataset_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for seq_idx, seq_dir in enumerate(sequence_dirs(dataset_dir)):
            video_name = seq_dir.parent.name
            seq_name = seq_dir.name
            frame0_path = find_frame_path(seq_dir, "I0")
            frame2_path = find_frame_path(seq_dir, "I2")
            frame0 = load_rgb(frame0_path)
            frame2 = load_rgb(frame2_path)
            if frame0.shape[:2] != frame2.shape[:2]:
                frame2 = resize_rgb(frame2, (frame0.shape[1], frame0.shape[0]))
            frame1 = interpolate_middle_frame(ppl, frame0, frame2)
            for frame_idx, image in enumerate([frame0, frame1, frame2]):
                suffix = frame0_path.suffix.lower()
                dst_name = f"{seq_idx:02d}_{frame_idx}_{video_name}_{seq_name}{suffix}"
                save_rgb(out_dir / dst_name, image)


def main() -> None:
    export_hackview_uprnet()
    print(f"Exported UPR-Net hackview to {HACKVIEW_UPR_ROOT}")


if __name__ == "__main__":
    main()
