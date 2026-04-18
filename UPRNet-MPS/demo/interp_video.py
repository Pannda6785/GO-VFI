import argparse
import math
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from core.pipeline import Pipeline
from core.device import resolve_device

DEVICE = resolve_device()


def infer_middle_frame(ppl, frame0, frame1):
    img0 = (torch.tensor(frame0.transpose(2, 0, 1)).to(DEVICE) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(frame1.transpose(2, 0, 1)).to(DEVICE) / 255.0).unsqueeze(0)

    _, _, h, w = img0.shape
    pyr_level = math.ceil(math.log2(w / 448) + 3)
    divisor = 2 ** (pyr_level - 1 + 2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    interp_img, _ = ppl.inference(img0, img1, time_period=0.5, pyr_level=pyr_level)
    interp_img = interp_img[:, :, :h, :w]
    return (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser(description="Interpolate a video with UPR-Net at 2x fps.")
    parser.add_argument("--input", required=True, help="Path to input video.")
    parser.add_argument("--output", required=True, help="Path to output video.")
    parser.add_argument("--frames_dir", required=True, help="Directory to save output frames.")
    parser.add_argument("--model_size", default="base", help="One of: base, large, LARGE.")
    parser.add_argument("--model_file", default="./checkpoints/upr-base.pkl", help="Path to model weights.")
    parser.add_argument("--device", default="auto", help="runtime device: auto, mps, cpu, cuda, or cuda:N")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frames_dir = Path(args.frames_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Failed to open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Invalid video metadata from OpenCV.")

    ok, prev_frame = capture.read()
    if not ok:
        raise SystemExit("Input video contains no readable frames.")

    model_cfg_dict = {
        "load_pretrain": True,
        "model_size": args.model_size,
        "model_file": args.model_file,
        "device": args.device,
    }
    global DEVICE
    DEVICE = resolve_device(args.device)
    torch.set_grad_enabled(False)
    ppl = Pipeline(model_cfg_dict)
    ppl.eval()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps * 2.0, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open output video for writing: {output_path}")

    output_index = 0
    writer.write(prev_frame)
    cv2.imwrite(str(frames_dir / f"{output_index:06d}.png"), prev_frame)
    output_index += 1
    frame_count = 1

    while True:
        ok, next_frame = capture.read()
        if not ok:
            break
        middle = infer_middle_frame(ppl, prev_frame, next_frame)
        writer.write(middle)
        cv2.imwrite(str(frames_dir / f"{output_index:06d}.png"), middle)
        output_index += 1
        writer.write(next_frame)
        cv2.imwrite(str(frames_dir / f"{output_index:06d}.png"), next_frame)
        output_index += 1
        prev_frame = next_frame
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} input frames...")

    capture.release()
    writer.release()
    print(f"Wrote interpolated video to {output_path}")
    print(f"Wrote {output_index} frames to {frames_dir}")


if __name__ == "__main__":
    main()
