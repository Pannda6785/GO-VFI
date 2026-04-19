import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from inference import (
    apply_mask_to_image,
    create_pure_mask,
    load_efficient_sam_model,
    run_efficient_sam_with_boxes,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tracked YOLO + EfficientSAM inference and group masks by persistent track ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--yolo-model", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--sam-type", type=str, default="vits", choices=["vitt", "vits"], help="EfficientSAM model type")
    parser.add_argument("--sam-model", type=str, required=True, help="Path to EfficientSAM weights (.pt or .pt.zip)")
    parser.add_argument("--source", type=str, required=True, help="Image file, directory of frames, or video path")

    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Ultralytics tracker config")
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="Class IDs to track")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[256, 448], help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16 in YOLO tracking")
    parser.add_argument("--agnostic-nms", action="store_true", help="Enable class-agnostic NMS")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")

    parser.add_argument("--project", type=str, default="runs/tracked_inference", help="Output project directory")
    parser.add_argument("--name", type=str, default="exp", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing output dir")
    parser.add_argument("--save-overlay", action="store_true", default=True, help="Save combined overlays")
    parser.add_argument("--no-save-overlay", dest="save_overlay", action="store_false", help="Disable combined overlays")
    return parser.parse_args()


def resolve_image_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    raise ValueError(f"Unsupported source for tracked image inference: {source}")


def annotate_track_detection(image, boxes, confidences, class_ids, track_ids, class_names, line_width=2):
    annotated = image.copy()
    for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
        class_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
        label = f"id={int(track_id)} {class_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return annotated


def annotate_track_overlay(image, masks, boxes, confidences, class_ids, track_ids, class_names, alpha=0.4, line_width=2):
    overlay = apply_mask_to_image(image, masks, alpha)
    return annotate_track_detection(overlay, boxes, confidences, class_ids, track_ids, class_names, line_width)


def save_track_artifacts(track_dir: Path, frame_stem: str, mask: np.ndarray, frame_bgr: np.ndarray, box, track_id: int):
    track_dir.mkdir(parents=True, exist_ok=True)
    mask_uint8 = mask.astype(np.uint8) * 255
    cv2.imwrite(str(track_dir / f"{frame_stem}_mask.png"), mask_uint8)

    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size > 0:
        cv2.imwrite(str(track_dir / f"{frame_stem}_crop.jpg"), crop)

    metadata_path = track_dir / "track_metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "track_id": int(track_id),
            "frame": frame_stem,
            "box_xyxy": [float(v) for v in box.tolist()],
            "mask_file": f"{frame_stem}_mask.png",
            "crop_file": f"{frame_stem}_crop.jpg" if crop.size > 0 else None,
        }) + "\n")


def process_tracked_images(image_paths, yolo_model, sam_model, args, device, output_dir: Path):
    frames_dir = output_dir / "frames"
    tracks_dir = output_dir / "tracks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(args.source),
        "tracker": args.tracker,
        "frames": [],
    }

    for image_path in image_paths:
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        results = yolo_model.track(
            frame_bgr,
            persist=True,
            classes=args.classes if args.classes else None,
            tracker=args.tracker,
            imgsz=args.imgsz,
            iou=args.iou,
            conf=args.conf,
            agnostic_nms=args.agnostic_nms,
            half=args.half,
            device=args.device if args.device else None,
            verbose=False,
        )
        result = results[0]
        frame_stem = image_path.stem
        frame_dir = frames_dir / frame_stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        if result.boxes is None or len(result.boxes) == 0:
            cv2.imwrite(str(frame_dir / f"{frame_stem}_original.jpg"), frame_bgr)
            summary["frames"].append({"frame": image_path.name, "detections": 0, "tracks": []})
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes), dtype=int)

        masks, mask_ious = run_efficient_sam_with_boxes(sam_model, frame_bgr, boxes, device)
        class_names = yolo_model.names if hasattr(yolo_model, "names") else {}

        pure_mask = create_pure_mask(frame_bgr.shape, masks)
        masked_image = apply_mask_to_image(frame_bgr, masks, args.alpha)
        detection_image = annotate_track_detection(
            frame_bgr,
            boxes,
            confidences,
            class_ids,
            track_ids,
            class_names,
            args.line_width,
        )
        combined_image = annotate_track_overlay(
            frame_bgr,
            masks,
            boxes,
            confidences,
            class_ids,
            track_ids,
            class_names,
            args.alpha,
            args.line_width,
        )

        cv2.imwrite(str(frame_dir / f"{frame_stem}_original.jpg"), frame_bgr)
        cv2.imwrite(str(frame_dir / f"{frame_stem}_mask.png"), pure_mask)
        cv2.imwrite(str(frame_dir / f"{frame_stem}_masked.jpg"), masked_image)
        cv2.imwrite(str(frame_dir / f"{frame_stem}_detection.jpg"), detection_image)
        if args.save_overlay:
            cv2.imwrite(str(frame_dir / f"{frame_stem}_combined.jpg"), combined_image)

        frame_tracks = []
        for box, conf, cls_id, track_id, mask, mask_iou in zip(boxes, confidences, class_ids, track_ids, masks, mask_ious):
            track_dir = tracks_dir / f"track_{int(track_id):04d}"
            save_track_artifacts(track_dir, frame_stem, mask, frame_bgr, box, int(track_id))
            frame_tracks.append({
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "confidence": float(conf),
                "mask_iou": float(mask_iou),
                "track_dir": str(track_dir),
            })

        summary["frames"].append({
            "frame": image_path.name,
            "detections": int(len(boxes)),
            "tracks": frame_tracks,
        })
        print(f"Processed {image_path.name}: {len(boxes)} tracked overlays")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.project) / args.name
    if output_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        if args.device.isdigit():
            device = torch.device(f"cuda:{args.device}")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Tracked GOSeg Inference")
    print("=" * 70)
    print(f"YOLO model: {args.yolo_model}")
    print(f"EfficientSAM model: {args.sam_model}")
    print(f"Source: {source}")
    print(f"Tracker: {args.tracker}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    yolo_model = YOLO(args.yolo_model)
    sam_model = load_efficient_sam_model(args.sam_type, args.sam_model, device)

    image_paths = resolve_image_paths(source)
    if not image_paths:
        raise ValueError(f"No supported images found in {source}")

    process_tracked_images(image_paths, yolo_model, sam_model, args, device, output_dir)
    print(f"Saved tracked masks and frame outputs to: {output_dir}")


if __name__ == "__main__":
    main()
