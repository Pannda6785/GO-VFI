import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with YOLO model for graphical overlay segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model weights (.pt file)"
    )
    
    # Input configuration
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (image file, video file, directory, or camera index)"
    )
    
    # Inference parameters
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[256, 448],
        help="Inference image size (height width) or single value for square"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for inference (e.g., '0' or 'cpu')"
    )
    
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half-precision inference"
    )
    
    # Visualization options
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save inference results"
    )
    
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Don't save inference results"
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results to text files"
    )
    
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidences in text files"
    )
    
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped prediction boxes"
    )
    
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide labels in visualizations"
    )
    
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence scores in visualizations"
    )
    
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Bounding box line width (pixels)"
    )
    
    # Output configuration
    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing experiment"
    )
    
    # Advanced options
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum number of detections per image"
    )
    
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Class-agnostic NMS"
    )
    
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Use high-resolution segmentation masks"
    )
    
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Video frame-rate stride (process every nth frame)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream inference (for large batches or videos)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def print_results_summary(results):
    """Print summary of inference results."""
    print("\n" + "=" * 70)
    print("Inference Summary")
    print("=" * 70)
    
    total_detections = 0
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            total_detections += len(r.boxes)
    
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    
    if len(results) > 0:
        avg_speed = np.mean([r.speed['inference'] for r in results if hasattr(r, 'speed')])
        print(f"Average inference time: {avg_speed:.2f}ms per image")
    
    print("=" * 70)


def run_batch_inference(model, source_path, args):
    """Run inference on a batch of images."""
    print(f"Running batch inference on: {source_path}")
    
    # Get list of images
    if source_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        print(f"Found {len(image_files)} images")
    else:
        image_files = [source_path]
    
    if not image_files:
        print("No valid images found!")
        return
    
    # Prepare prediction arguments
    predict_args = {
        "source": str(source_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "half": args.half,
        "show": args.show,
        "save": args.save,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
        "hide_labels": args.hide_labels,
        "hide_conf": args.hide_conf,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "max_det": args.max_det,
        "agnostic_nms": args.agnostic_nms,
        "retina_masks": args.retina_masks,
        "vid_stride": args.vid_stride,
        "stream": args.stream,
        "verbose": args.verbose,
    }
    
    # Add optional parameters
    if args.device:
        predict_args["device"] = args.device
    if args.line_width:
        predict_args["line_width"] = args.line_width
    
    # Run inference
    results = model.predict(**predict_args)
    
    return results


def run_camera_inference(model, camera_idx, args):
    """Run real-time inference on camera feed."""
    print(f"Starting camera inference (index: {camera_idx})")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_idx}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run inference
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else None,
                half=args.half,
                max_det=args.max_det,
                agnostic_nms=args.agnostic_nms,
                retina_masks=args.retina_masks,
                verbose=False,
            )
            
            # Visualize results
            annotated_frame = results[0].plot(
                line_width=args.line_width,
                labels=not args.hide_labels,
                conf=not args.hide_conf,
            )
            
            # Display
            cv2.imshow('YOLO Inference', annotated_frame)
            
            frame_count += 1
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("YOLO Graphical Overlay Segmentation Inference")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Half precision: {args.half}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 70)
    
    # Verify model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Load model
    print(f"\nLoading YOLO model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded successfully!")
    
    # Check if source is camera index
    try:
        camera_idx = int(args.source)
        # If source is a number, treat as camera index
        run_camera_inference(model, camera_idx, args)
        return
    except ValueError:
        # Not a camera index, continue with file/directory
        pass
    
    # Check source type
    source_path = Path(args.source)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    
    # Run inference
    print("\nStarting inference...\n")
    results = run_batch_inference(model, source_path, args)
    
    # Print summary
    if results and not args.stream:
        results_list = list(results) if hasattr(results, '__iter__') else [results]
        print_results_summary(results_list)
        
        if args.save:
            save_dir = Path(args.project) / args.name
            print(f"\nResults saved to: {save_dir}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
