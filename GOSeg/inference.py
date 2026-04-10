import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from torchvision import transforms
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import zipfile


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run combined YOLO + EfficientSAM inference for graphical overlay segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="./weights/yolo_best.pt",
        help="Path to trained YOLO detection model weights (.pt file)"
    )
    
    parser.add_argument(
        "--sam-type",
        type=str,
        default="vitt",
        choices=["vitt", "vits"],
        help="EfficientSAM model type (vitt=tiny, vits=small)"
    )
    
    parser.add_argument(
        "--sam-model",
        type=str,
        default="./weights/efficient_sam_vitt.pt",
        help="Path to EfficientSAM model weights (.pt or .pt.zip file)"
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
        default=1024,
        help="Inference image size for EfficientSAM (not used, kept for compatibility)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO detections"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="IoU threshold for NMS (not used by EfficientSAM, kept for compatibility)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for inference (e.g., '0' or 'cpu')"
    )
    
    parser.add_argument(
        "--retina",
        action="store_true",
        help="Not used with EfficientSAM (kept for compatibility)"
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
        "--save-masks",
        action="store_true",
        help="Save individual mask files separately (in individual_masks subfolder)"
    )
    
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save combined visualization (masks + boxes together)"
    )
    
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width (pixels)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparency for mask overlay (0.0-1.0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--project",
        type=str,
        default="runs/combined_inference",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
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
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Show timing information for each stage"
    )
    
    return parser.parse_args()


def get_boxes_from_yolo(yolo_model, image, conf_threshold, device):
    """
    Run YOLO detection to get bounding boxes.
    
    Args:
        yolo_model: YOLO model instance
        image: Input image (numpy array)
        conf_threshold: Confidence threshold
        device: Device to run on
        
    Returns:
        boxes: Detected bounding boxes in xyxy format
        confidences: Confidence scores
        class_ids: Class IDs
    """
    # Run YOLO detection
    results = yolo_model.predict(
        image,
        conf=conf_threshold,
        device=device,
        verbose=False
    )
    
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None, None
    
    # Extract boxes, confidences, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    return boxes, confidences, class_ids


def run_efficient_sam_with_boxes(sam_model, image, boxes, device):
    """
    Run EfficientSAM with box prompts from YOLO detections.
    Uses box encoding directly for efficient segmentation.
    Processes all boxes in a single batched inference call.
    
    Args:
        sam_model: EfficientSAM model instance
        image: Input image (numpy array, BGR format)
        boxes: Bounding boxes from YOLO in xyxy format
        device: Device to run on
        
    Returns:
        masks: Segmentation masks (list of numpy arrays)
        predicted_ious: IoU scores for each mask
    """
    # Get original image dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    image_tensor = transforms.ToTensor()(image_rgb)
    
    # Move to device
    image_tensor = image_tensor.to(device)
    sam_model = sam_model.to(device)
    
    # Prepare all boxes for batched inference
    num_boxes = len(boxes)
    all_points = []
    valid_indices = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, orig_w - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        y2 = max(0, min(y2, orig_h - 1))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Add box corner points (top-left and bottom-right)
        all_points.append([[x1, y1], [x2, y2]])
        valid_indices.append(i)
    
    # Handle case with no valid boxes
    if len(all_points) == 0:
        return [np.zeros((orig_h, orig_w), dtype=bool) for _ in range(num_boxes)], [0.0] * num_boxes
    
    # Convert to tensors - shape: [1, num_valid_boxes, 2, 2]
    input_points = torch.tensor([all_points], dtype=torch.float32).to(device)
    # Labels [2, 3] indicate box corners - shape: [1, num_valid_boxes, 2]
    input_labels = torch.tensor([[[2, 3]] * len(all_points)]).to(device)
    
    try:
        # Run EfficientSAM inference on ALL boxes at once
        with torch.no_grad():
            predicted_logits, predicted_iou = sam_model(
                image_tensor[None, ...],
                input_points,
                input_labels,
            )
        
        # Extract masks for each box
        selected_masks = [None] * num_boxes
        predicted_ious = [0.0] * num_boxes
        
        for i, box_idx in enumerate(valid_indices):
            # Sort by IoU and get best mask for this box
            sorted_ids = torch.argsort(predicted_iou[0, i], dim=-1, descending=True)
            best_mask_idx = sorted_ids[0]
            
            # Get the mask with highest IoU
            mask = torch.ge(predicted_logits[0, i, best_mask_idx, :, :], 0).cpu().detach().numpy()
            iou = predicted_iou[0, i, best_mask_idx].cpu().item()
            
            selected_masks[box_idx] = mask.astype(bool)
            predicted_ious[box_idx] = iou
        
        # Fill in invalid boxes with empty masks
        for i in range(num_boxes):
            if selected_masks[i] is None:
                selected_masks[i] = np.zeros((orig_h, orig_w), dtype=bool)
        
        return selected_masks, predicted_ious
        
    except Exception as e:
        print(f"Warning: EfficientSAM batched inference failed: {e}")
        print("Falling back to per-box processing...")
        
        # Fallback: create simple box masks
        selected_masks = []
        predicted_ious = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(float)
            x1 = max(0, min(int(x1), orig_w - 1))
            x2 = max(0, min(int(x2), orig_w - 1))
            y1 = max(0, min(int(y1), orig_h - 1))
            y2 = max(0, min(int(y2), orig_h - 1))
            
            mask = np.zeros((orig_h, orig_w), dtype=bool)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
            
            selected_masks.append(mask)
            predicted_ious.append(0.0)
        
        return selected_masks, predicted_ious


def load_efficient_sam_model(model_type, model_path, device):
    """
    Load EfficientSAM model.
    
    Args:
        model_type: Model type ('vitt' or 'vits')
        model_path: Path to model weights
        device: Device to load model on
        
    Returns:
        model: Loaded EfficientSAM model
    """
    model_path = Path(model_path)
    
    # Handle .zip files
    if model_path.suffix == '.zip' or str(model_path).endswith('.pt.zip'):
        print(f"Extracting model from zip file: {model_path}")
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            # Extract to the same directory
            zip_ref.extractall(model_path.parent)
        # Update path to extracted file
        model_path = model_path.parent / model_path.stem
        if model_path.suffix == '.pt':
            # Already has .pt extension
            pass
        else:
            # Add .pt extension
            model_path = Path(str(model_path) + '.pt')
    
    if not model_path.exists():
        raise FileNotFoundError(f"EfficientSAM model file not found: {model_path}")
    
    # Build model based on type
    if model_type == 'vitt':
        model = build_efficient_sam_vitt(checkpoint=str(model_path))
    elif model_type == 'vits':
        model = build_efficient_sam_vits(checkpoint=str(model_path))
    else:
        raise ValueError(f"Unknown EfficientSAM model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_colors(num_detections, seed=42):
    """Generate consistent colors for detections."""
    np.random.seed(seed)
    return np.random.randint(0, 255, size=(num_detections, 3), dtype=np.uint8)


def create_pure_mask(image_shape, masks):
    """
    Create pure binary mask visualization (white masks on black background).
    
    Args:
        image_shape: Shape of the original image (H, W, C)
        masks: List of segmentation masks
        
    Returns:
        mask_image: Binary mask image
    """
    h, w = image_shape[:2]
    mask_image = np.zeros((h, w), dtype=np.uint8)
    
    if masks is not None and len(masks) > 0:
        for mask in masks:
            if isinstance(mask, np.ndarray):
                binary_mask = mask.astype(bool)
            else:
                binary_mask = mask.cpu().numpy().astype(bool)
            
            mask_image[binary_mask] = 255
    
    return mask_image


def apply_mask_to_image(image, masks, alpha=0.4):
    """
    Apply colored masks to the image without bounding boxes.
    
    Args:
        image: Original image
        masks: List of segmentation masks
        alpha: Mask transparency
        
    Returns:
        masked_image: Image with colored masks applied
    """
    masked_image = image.copy()
    
    if masks is not None and len(masks) > 0:
        colors = get_colors(len(masks))
        
        for i, mask in enumerate(masks):
            color = colors[i].tolist()
            
            # Convert mask to binary
            if isinstance(mask, np.ndarray):
                binary_mask = mask.astype(bool)
            else:
                binary_mask = mask.cpu().numpy().astype(bool)
            
            # Apply colored mask
            for c in range(3):
                masked_image[:, :, c] = np.where(
                    binary_mask,
                    masked_image[:, :, c] * (1 - alpha) + alpha * color[c],
                    masked_image[:, :, c]
                )
    
    return masked_image


def create_detection_output(image, boxes, confidences, class_ids, class_names, line_width=2):
    """
    Create detection visualization with only bounding boxes and labels.
    
    Args:
        image: Original image
        boxes: YOLO bounding boxes
        confidences: Detection confidences
        class_ids: Class IDs
        class_names: Class names dictionary
        line_width: Box line width
        
    Returns:
        detection_image: Image with bounding boxes and labels
    """
    detection_image = image.copy()
    
    if boxes is not None and len(boxes) > 0:
        colors = get_colors(len(boxes))
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i].tolist()
            
            # Draw box
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), color, line_width)
            
            # Draw label
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            label = f"{class_name} {conf:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                detection_image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                detection_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return detection_image


def visualize_results(image, boxes, masks, confidences, class_ids, class_names, 
                     alpha=0.4, line_width=2):
    """
    Visualize combined YOLO + EfficientSAM results (masks + bounding boxes).
    
    Args:
        image: Original image
        boxes: YOLO bounding boxes
        masks: EfficientSAM segmentation masks
        confidences: Detection confidences
        class_ids: Class IDs
        class_names: Class names dictionary
        alpha: Mask transparency
        line_width: Box line width
        
    Returns:
        vis_image: Visualization image
    """
    vis_image = image.copy()
    
    # Generate colors for each detection
    colors = get_colors(len(boxes))
    
    # Apply masks
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            if i >= len(colors):
                break
                
            color = colors[i].tolist()
            
            # Convert mask to binary
            if isinstance(mask, np.ndarray):
                binary_mask = mask.astype(bool)
            else:
                binary_mask = mask.cpu().numpy().astype(bool)
            
            # Apply colored mask
            for c in range(3):
                vis_image[:, :, c] = np.where(
                    binary_mask,
                    vis_image[:, :, c] * (1 - alpha) + alpha * color[c],
                    vis_image[:, :, c]
                )
    
    # Draw bounding boxes and labels
    if boxes is not None and len(boxes) > 0:
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i].tolist()
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_width)
            
            # Draw label
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            label = f"{class_name} {conf:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return vis_image


def process_image(image_path, yolo_model, sam_model, args, device, save_dir=None):
    """
    Process a single image with YOLO + EfficientSAM.
    
    Args:
        image_path: Path to image file
        yolo_model: YOLO model
        sam_model: EfficientSAM model
        args: CLI arguments
        device: Torch device
        save_dir: Directory to save results
        
    Returns:
        vis_image: Visualization image
        stats: Processing statistics
    """
    stats = {
        'yolo_time': 0,
        'sam_time': 0,
        'total_time': 0,
        'num_detections': 0
    }
    
    start_total = time.time()
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None, stats
    
    original_image = image.copy()
    
    # Step 1: YOLO Detection
    if args.benchmark:
        start = time.time()
    
    boxes, confidences, class_ids = get_boxes_from_yolo(
        yolo_model, image, args.conf, args.device if args.device else 'cpu'
    )
    
    if args.benchmark:
        stats['yolo_time'] = time.time() - start
    
    if boxes is None or len(boxes) == 0:
        print(f"No detections found in {image_path.name}")
        stats['total_time'] = time.time() - start_total
        return original_image, stats
    
    stats['num_detections'] = len(boxes)
    
    # Step 2: EfficientSAM with box prompts (direct box encoding)
    if args.benchmark:
        start = time.time()
    
    masks, predicted_ious = run_efficient_sam_with_boxes(
        sam_model, image, boxes, device
    )
    
    if args.benchmark:
        stats['sam_time'] = time.time() - start
    
    # Step 3: Visualize results
    class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
    
    # Create combined visualization (masks + boxes)
    vis_image = visualize_results(
        original_image, boxes, masks, confidences, class_ids,
        class_names, args.alpha, args.line_width
    )
    
    stats['total_time'] = time.time() - start_total
    
    # Save results
    if save_dir and args.save:
        image_dir = save_dir / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save pure binary mask (white on black)
        pure_mask = create_pure_mask(original_image.shape, masks)
        mask_path = image_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), pure_mask)
        
        # 2. Save mask applied to image (colored masks only, no boxes)
        masked_image = apply_mask_to_image(original_image, masks, args.alpha)
        masked_path = image_dir / f"{image_path.stem}_masked.jpg"
        cv2.imwrite(str(masked_path), masked_image)
        
        # 3. Save detection output (boxes and labels only, no masks)
        detection_image = create_detection_output(
            original_image, boxes, confidences, class_ids,
            class_names, args.line_width
        )
        detection_path = image_dir / f"{image_path.stem}_detection.jpg"
        cv2.imwrite(str(detection_path), detection_image)
        
        # 4. Save combined visualization (optional, if save_overlay is True)
        if args.save_overlay:
            combined_path = image_dir / f"{image_path.stem}_combined.jpg"
            cv2.imwrite(str(combined_path), vis_image)
        
        # Save individual masks if requested
        if args.save_masks and masks is not None:
            masks_dir = image_dir / "individual_masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            for i, mask in enumerate(masks):
                if isinstance(mask, np.ndarray):
                    mask_img = (mask * 255).astype(np.uint8)
                else:
                    mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                
                mask_path = masks_dir / f"mask_{i}.png"
                cv2.imwrite(str(mask_path), mask_img)
    
    return vis_image, stats


def process_video(video_path, yolo_model, sam_model, args, device, save_dir=None):
    """
    Process a video file with YOLO + EfficientSAM.
    
    Args:
        video_path: Path to video file
        yolo_model: YOLO model
        sam_model: EfficientSAM model
        args: CLI arguments
        device: Torch device
        save_dir: Directory to save results
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Setup video writer
    writer = None
    writer_masked = None
    writer_detection = None
    writer_combined = None
    
    if save_dir and args.save:
        output_dir = save_dir / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Writer for mask applied to image
        masked_path = output_dir / f"{video_path.stem}_masked.mp4"
        writer_masked = cv2.VideoWriter(str(masked_path), fourcc, fps, (width, height))
        
        # Writer for detection output
        detection_path = output_dir / f"{video_path.stem}_detection.mp4"
        writer_detection = cv2.VideoWriter(str(detection_path), fourcc, fps, (width, height))
        
        # Writer for combined output (if enabled)
        if args.save_overlay:
            combined_path = output_dir / f"{video_path.stem}_combined.mp4"
            writer_combined = cv2.VideoWriter(str(combined_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            boxes, confidences, class_ids = get_boxes_from_yolo(
                yolo_model, frame, args.conf, args.device if args.device else 'cpu'
            )
            
            if boxes is not None and len(boxes) > 0:
                total_detections += len(boxes)
                
                # Run EfficientSAM
                masks, _ = run_efficient_sam_with_boxes(
                    sam_model, frame, boxes, device
                )
                
                # Create visualizations
                class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
                
                # Masked image (masks only)
                masked_frame = apply_mask_to_image(frame, masks, args.alpha)
                
                # Detection output (boxes only)
                detection_frame = create_detection_output(
                    frame, boxes, confidences, class_ids,
                    class_names, args.line_width
                )
                
                # Combined (masks + boxes)
                vis_frame = visualize_results(
                    frame, boxes, masks, confidences, class_ids,
                    class_names, args.alpha, args.line_width
                )
            else:
                masked_frame = frame
                detection_frame = frame
                vis_frame = frame
            
            # Save frames
            if writer_masked:
                writer_masked.write(masked_frame)
            if writer_detection:
                writer_detection.write(detection_frame)
            if writer_combined:
                writer_combined.write(vis_frame)
            
            # Show frame
            if args.show:
                cv2.imshow('YOLO + EfficientSAM Inference', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
    
    finally:
        cap.release()
        if writer_masked:
            writer_masked.release()
        if writer_detection:
            writer_detection.release()
        if writer_combined:
            writer_combined.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print(f"Video processing complete: {frame_count} frames, {total_detections} total detections")
    
    if save_dir and args.save:
        output_dir = save_dir / video_path.stem
        print(f"\nSaved video outputs to: {output_dir}")
        print("  - *_masked.mp4: Mask applied to video")
        print("  - *_detection.mp4: Bounding boxes and labels")
        if args.save_overlay:
            print("  - *_combined.mp4: Combined masks and boxes")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("YOLO + EfficientSAM Combined Inference")
    print("=" * 70)
    print(f"YOLO Model: {args.yolo_model}")
    print(f"EfficientSAM Model: {args.sam_model}")
    print(f"EfficientSAM Type: {args.sam_type}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Mask alpha: {args.alpha}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 70)
    
    # Verify model files exist
    yolo_path = Path(args.yolo_model)
    sam_path = Path(args.sam_model)
    
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model file not found: {args.yolo_model}")
    if not sam_path.exists():
        raise FileNotFoundError(f"EfficientSAM model file not found: {args.sam_model}")
    
    # Setup device
    if args.device:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load models
    print(f"\nLoading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    print("YOLO model loaded!")
    
    print(f"Loading EfficientSAM model: {args.sam_model} (type: {args.sam_type})")
    sam_model = load_efficient_sam_model(args.sam_type, args.sam_model, device)
    print("EfficientSAM model loaded!")
    
    # Setup save directory
    save_dir = None
    if args.save:
        save_dir = Path(args.project) / args.name
        
        # Handle existing directory
        if save_dir.exists() and not args.exist_ok:
            # Find next available number
            counter = 1
            while (Path(args.project) / f"{args.name}{counter}").exists():
                counter += 1
            save_dir = Path(args.project) / f"{args.name}{counter}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
    
    # Check source type
    source_path = Path(args.source)
    
    # Check if source is camera index
    if args.source.isdigit():
        print("\nCamera input not yet implemented for combined inference")
        return
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    
    # Process based on source type
    print("\nStarting inference...\n")
    
    if source_path.is_file():
        # Check if video or image
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        if source_path.suffix.lower() in video_extensions:
            # Process video
            process_video(source_path, yolo_model, sam_model, args, device, save_dir)
        elif source_path.suffix.lower() in image_extensions:
            # Process single image
            vis_image, stats = process_image(
                source_path, yolo_model, sam_model, args, device, save_dir
            )
            
            if vis_image is not None:
                print(f"\nResults for {source_path.name}:")
                print(f"  Detections: {stats['num_detections']}")
                if args.benchmark:
                    print(f"  YOLO time: {stats['yolo_time']:.3f}s")
                    print(f"  EfficientSAM time: {stats['sam_time']:.3f}s")
                    print(f"  Total time: {stats['total_time']:.3f}s")
                
                if args.save and save_dir:
                    img_dir = save_dir / source_path.stem
                    print(f"\n  Saved outputs:")
                    print(f"    - Binary mask: {img_dir / f'{source_path.stem}_mask.png'}")
                    print(f"    - Masked image: {img_dir / f'{source_path.stem}_masked.jpg'}")
                    print(f"    - Detection: {img_dir / f'{source_path.stem}_detection.jpg'}")
                    if args.save_overlay:
                        print(f"    - Combined: {img_dir / f'{source_path.stem}_combined.jpg'}")
                
                if args.show:
                    cv2.imshow('Result', vis_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"Unsupported file format: {source_path.suffix}")
            return
    
    elif source_path.is_dir():
        # Process directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No valid images found in directory!")
            return
        
        print(f"Found {len(image_files)} images")
        
        all_stats = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")
            
            vis_image, stats = process_image(
                img_path, yolo_model, sam_model, args, device, save_dir
            )
            
            if vis_image is not None:
                all_stats.append(stats)
                print(f"  Detections: {stats['num_detections']}")
                if args.benchmark:
                    print(f"  Time: {stats['total_time']:.3f}s")
        
        # Print summary
        if all_stats:
            print("\n" + "=" * 70)
            print("Inference Summary")
            print("=" * 70)
            print(f"Total images processed: {len(all_stats)}")
            print(f"Total detections: {sum(s['num_detections'] for s in all_stats)}")
            if args.benchmark:
                avg_yolo = np.mean([s['yolo_time'] for s in all_stats])
                avg_sam = np.mean([s['sam_time'] for s in all_stats])
                avg_total = np.mean([s['total_time'] for s in all_stats])
                print(f"Average YOLO time: {avg_yolo:.3f}s")
                print(f"Average EfficientSAM time: {avg_sam:.3f}s")
                print(f"Average total time: {avg_total:.3f}s")
            print("\nFor each image, saved:")
            print("  - *_mask.png: Pure binary mask (white on black)")
            print("  - *_masked.jpg: Mask applied to original image")
            print("  - *_detection.jpg: Bounding boxes and labels")
            if args.save_overlay:
                print("  - *_combined.jpg: Combined masks and boxes")
            print("=" * 70)
    
    if save_dir:
        print(f"\nResults saved to: {save_dir}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
