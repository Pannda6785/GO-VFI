#!/usr/bin/env python3
"""Build the DAVIS triplet mini-project datasets from isolated configs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CatalogEntry:
    cls_name: str
    subfolder: str
    directory: Path
    files: list[Path]
    weight: float


@dataclass
class OverlayAsset:
    cls_name: str
    source_rel: str
    image: np.ndarray
    alpha: np.ndarray
    meta: dict


@dataclass
class Proposal:
    score: float
    positions: list[tuple[int, int]]
    bboxes: list[tuple[int, int, int, int]]
    crop_boxes: list[tuple[int, int, int, int]]
    masks: list[np.ndarray]
    mask_areas: list[int]


@dataclass
class PreparedOverlay:
    overlay_id: int
    asset: OverlayAsset
    frame_images: list[np.ndarray]
    frame_alphas: list[np.ndarray]
    proposal_frame_images: list[np.ndarray]
    proposal_frame_alphas: list[np.ndarray]
    mode: str
    geometry_mode: str
    geometry_params: dict
    motion_step: tuple[float, float]
    alpha_scales: list[float]
    color: int
    proposals: list[Proposal]


@dataclass
class SequenceItem:
    video_name: str
    seq_name: str
    path: Path


cfg = None


def load_config(config_path: Path):
    module_name = f"davis_triplet_mini_cfg_{config_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(config_path.parent.parent.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def weighted_choice(items, weights, rng: random.Random):
    total = float(sum(weights))
    pick = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += float(weight)
        if pick <= acc:
            return item
    return items[-1]


def weighted_sample_without_replacement(items, weights, k: int, rng: random.Random):
    pool_items = list(items)
    pool_weights = list(weights)
    chosen = []
    for _ in range(min(k, len(pool_items))):
        item = weighted_choice(pool_items, pool_weights, rng)
        idx = pool_items.index(item)
        chosen.append(item)
        pool_items.pop(idx)
        pool_weights.pop(idx)
    return chosen


def load_catalog() -> dict[str, list[CatalogEntry]]:
    by_class: dict[str, list[CatalogEntry]] = {}
    lines = [line.strip() for line in cfg.TRAIN_SPLIT.read_text().splitlines() if line.strip()]
    for rel in lines:
        directory = cfg.GOONS_ROOT / rel
        if not directory.exists():
            continue
        cls_name = rel.split("/", 1)[0]
        if cls_name == "Font":
            files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}]
        else:
            files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue
        weight = 1.0 + len(files) ** (2.0 / 3.0)
        by_class.setdefault(cls_name, []).append(
            CatalogEntry(
                cls_name=cls_name,
                subfolder=rel,
                directory=directory,
                files=sorted(files),
                weight=weight,
            )
        )
    return by_class


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_rgba_asset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if alpha.max() > 0:
        return rgb, alpha

    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0).astype(np.float32)
    bg = corners.mean(axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    alpha = np.where(dist > 18.0, 255, 0).astype(np.uint8)
    if alpha.max() == 0:
        alpha.fill(255)
    return rgb, alpha


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path, quality=95)


def save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)


def resize_overlay(image: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    rgb_img = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
    alpha_img = Image.fromarray(alpha, mode="L").resize((width, height), Image.Resampling.LANCZOS)
    return np.array(rgb_img, dtype=np.uint8), np.array(alpha_img, dtype=np.uint8)


def center_on_canvas(image: np.ndarray, alpha: np.ndarray, canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    canvas_w, canvas_h = canvas_size
    src_h, src_w = image.shape[:2]
    out_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    out_alpha = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    src_y0 = max(0, (src_h - canvas_h) // 2)
    src_x0 = max(0, (src_w - canvas_w) // 2)
    src_y1 = src_y0 + min(src_h, canvas_h)
    src_x1 = src_x0 + min(src_w, canvas_w)
    cropped_img = image[src_y0:src_y1, src_x0:src_x1]
    cropped_alpha = alpha[src_y0:src_y1, src_x0:src_x1]
    y0 = max(0, (canvas_h - cropped_img.shape[0]) // 2)
    x0 = max(0, (canvas_w - cropped_img.shape[1]) // 2)
    out_img[y0 : y0 + cropped_img.shape[0], x0 : x0 + cropped_img.shape[1]] = cropped_img
    out_alpha[y0 : y0 + cropped_alpha.shape[0], x0 : x0 + cropped_alpha.shape[1]] = cropped_alpha
    return out_img, out_alpha


def alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def alpha_centroid(alpha: np.ndarray) -> tuple[float, float]:
    weights = alpha.astype(np.float32)
    total = float(weights.sum())
    if total <= 0.0:
        h, w = alpha.shape
        return (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.indices(alpha.shape, dtype=np.float32)
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    return cx, cy


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def random_text(rng: random.Random, length: int | None = None) -> str:
    length = length if length is not None else rng.randint(*cfg.font_text_length_range)
    return "".join(rng.choice(cfg.font_text_alphabet) for _ in range(length))


def random_text_same_length(original: str, rng: random.Random) -> str:
    generated = "".join(rng.choice(cfg.font_text_alphabet) if ch != " " else " " for ch in original)
    if generated == original:
        generated = generated[::-1]
    return generated


def render_font_asset(
    font_path: Path,
    rng: random.Random,
    canvas_size: tuple[int, int] | None = None,
    text: str | None = None,
    font_size: int | None = None,
    fill: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    text = text or random_text(rng)
    if fill is None:
        fill = (255, 255, 255) if rng.random() < cfg.p_white_text else rng.choice(cfg.font_fill_palette)
    stroke_width = stroke_width if stroke_width is not None else rng.randint(*cfg.font_stroke_width)
    stroke_fill = tuple(max(0, c - 160) for c in fill)
    target_w, target_h = canvas_size if canvas_size else (512, 160)
    padding = 10

    size = font_size if font_size is not None else rng.randint(*cfg.font_size_range)
    while size >= 12:
        font = ImageFont.truetype(str(font_path), size=size)
        probe = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        if (bbox[2] - bbox[0] + padding * 2) <= target_w and (bbox[3] - bbox[1] + padding * 2) <= target_h:
            break
        size -= 2

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    x = max(0, (target_w - (bbox[2] - bbox[0])) // 2 - bbox[0])
    y = max(0, (target_h - (bbox[3] - bbox[1])) // 2 - bbox[1])
    draw.text((x, y), text, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
    rgba = np.array(canvas, dtype=np.uint8)
    return rgba[..., :3], rgba[..., 3], {
        "text": text,
        "font_size": size,
        "fill": list(fill),
        "stroke_width": stroke_width,
    }


def apply_overlay_aug(image: np.ndarray, alpha: np.ndarray, replay: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if replay is None:
        augmented = cfg.overlay_aug(image=image, mask=alpha)
    else:
        augmented = cfg.overlay_aug.replay(replay, image=image, mask=alpha)
    return augmented["image"], augmented["mask"], augmented["replay"]


def sample_text_transition_mode(rng: random.Random) -> str:
    modes = list(cfg.appearance_change_text_transition_probs.keys())
    weights = list(cfg.appearance_change_text_transition_probs.values())
    return weighted_choice(modes, weights, rng)


def add_text_within_mask(
    base_image: np.ndarray,
    alpha: np.ndarray,
    font_path: Path,
    text: str,
    rng: random.Random,
    fill: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
) -> np.ndarray:
    bbox = alpha_bbox(alpha)
    if bbox is None:
        return base_image.copy()
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    if fill is None:
        fill = (255, 255, 255) if rng.random() < cfg.p_white_text else rng.choice(cfg.font_fill_palette)
    stroke_width = stroke_width if stroke_width is not None else rng.randint(*cfg.font_stroke_width)
    stroke_fill = tuple(max(0, c - 160) for c in fill)
    overlay = Image.fromarray(base_image.copy(), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    size = min(max(12, box_h), cfg.font_size_range[1])
    font = ImageFont.truetype(str(font_path), size=size)
    bbox_text = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    while size >= 10 and ((bbox_text[2] - bbox_text[0]) > box_w or (bbox_text[3] - bbox_text[1]) > box_h):
        size -= 2
        font = ImageFont.truetype(str(font_path), size=size)
        bbox_text = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_x = x0 + max(0, (box_w - (bbox_text[2] - bbox_text[0])) // 2 - bbox_text[0])
    text_y = y0 + max(0, (box_h - (bbox_text[3] - bbox_text[1])) // 2 - bbox_text[1])
    draw.text((text_x, text_y), text, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
    rendered = np.array(overlay, dtype=np.uint8)
    out = base_image.copy()
    mask = alpha > 0
    out[mask] = rendered[mask]
    return out


def apply_color_change(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    lo, hi = cfg.appearance_change_color_scale_range
    factors = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=np.float32)
    perm = np.array(rng.sample([0, 1, 2], 3))
    out = np.clip(image.astype(np.float32) * factors[None, None, :], 0, 255).astype(np.uint8)
    out = out[..., perm]
    return out, {"color_multipliers": factors.tolist(), "color_permutation": perm.tolist()}


def apply_color_scale_only(image: np.ndarray, rng: random.Random, lo: float, hi: float) -> np.ndarray:
    factors = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=np.float32)
    return np.clip(image.astype(np.float32) * factors[None, None, :], 0, 255).astype(np.uint8)


def sample_catalog_entry(catalog: dict[str, list[CatalogEntry]], cls_name: str, rng: random.Random) -> CatalogEntry:
    entries = catalog[cls_name]
    return weighted_choice(entries, [entry.weight for entry in entries], rng)


def sample_overlay_asset_for_class(catalog: dict[str, list[CatalogEntry]], cls_name: str, rng: random.Random) -> OverlayAsset:
    entry = sample_catalog_entry(catalog, cls_name, rng)
    source_path = rng.choice(entry.files)
    if cls_name == "Font":
        image, alpha, font_meta = render_font_asset(source_path, rng)
        return OverlayAsset(
            cls_name=cls_name,
            source_rel=str(source_path.relative_to(cfg.GOONS_ROOT)),
            image=image,
            alpha=alpha,
            meta={"subfolder": entry.subfolder, **font_meta},
        )
    image, alpha = load_rgba_asset(source_path)
    return OverlayAsset(
        cls_name=cls_name,
        source_rel=str(source_path.relative_to(cfg.GOONS_ROOT)),
        image=image,
        alpha=alpha,
        meta={"subfolder": entry.subfolder},
    )


def sample_font_paths(catalog: dict[str, list[CatalogEntry]]) -> list[Path]:
    paths: list[Path] = []
    for entry in catalog.get("Font", []):
        paths.extend(entry.files)
    return sorted(set(paths))


def affine_canvas_size(width: int, height: int, params: dict) -> tuple[int, int]:
    components = set(params.get("components", []))
    box_w = float(width)
    box_h = float(height)
    if "scale" in components:
        max_scale = max(1.0, abs(params.get("end_scale", 1.0)))
        box_w *= max_scale
        box_h *= max_scale
    if "squeeze" in components:
        box_w *= max(1.0, abs(params.get("end_scale_x", 1.0)))
        box_h *= max(1.0, abs(params.get("end_scale_y", 1.0)))
    if "shear" in components:
        max_shear = abs(params.get("shear_velocity", 0.0)) * (cfg.SEQUENCE_LENGTH - 1)
        box_w = box_w + max_shear * box_h
    if "rotate" in components:
        diag = math.hypot(box_w, box_h)
        box_w = diag
        box_h = diag
    return int(math.ceil(box_w)), int(math.ceil(box_h))


def sample_affine_params(rng: random.Random) -> dict:
    component_names = ["rotate", "scale", "squeeze", "shear"]
    num_components = 2 if rng.random() < cfg.affine_two_component_prob else 1
    chosen = weighted_sample_without_replacement(
        component_names,
        [cfg.affine_component_weights[name] for name in component_names],
        num_components,
        rng,
    )
    params = {"components": chosen}
    if "rotate" in chosen:
        params["angle_velocity"] = rng.uniform(*cfg.affine_angle_velocity_range)
    if "scale" in chosen:
        params["end_scale"] = rng.uniform(*cfg.affine_scale_end_range)
    if "squeeze" in chosen:
        squeeze = rng.uniform(*cfg.affine_squeeze_end_range)
        if rng.random() < 0.5:
            params["end_scale_x"] = squeeze
            params["end_scale_y"] = 1.0
        else:
            params["end_scale_x"] = 1.0
            params["end_scale_y"] = squeeze
    if "shear" in chosen:
        params["shear_velocity"] = rng.uniform(*cfg.affine_shear_velocity_range)
    return params


def temporal_forward_matrix(mode: str, params: dict, frame_idx: int, alpha: np.ndarray) -> np.ndarray:
    if mode != "affine":
        return np.eye(3, dtype=np.float32)
    progress = frame_idx / max(cfg.SEQUENCE_LENGTH - 1, 1)
    angle = params.get("angle_velocity", 0.0) * frame_idx
    scale_iso = 1.0 + (params.get("end_scale", 1.0) - 1.0) * progress
    scale_x = scale_iso * (1.0 + (params.get("end_scale_x", 1.0) - 1.0) * progress)
    scale_y = scale_iso * (1.0 + (params.get("end_scale_y", 1.0) - 1.0) * progress)
    shear = params.get("shear_velocity", 0.0) * frame_idx
    cx, cy = alpha_centroid(alpha)
    theta = math.radians(angle)
    rot = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    shear_m = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return back @ rot @ shear_m @ scale @ to_origin


def apply_temporal_transform(image: np.ndarray, alpha: np.ndarray, mode: str, params: dict, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if mode != "affine":
        return image, alpha
    rgb_img = Image.fromarray(image, mode="RGB")
    alpha_img = Image.fromarray(alpha, mode="L")
    base_w, base_h = image.shape[1], image.shape[0]
    forward = temporal_forward_matrix(mode, params, frame_idx, alpha)
    inverse = np.linalg.inv(forward)
    coeffs = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )
    transformed_img = np.array(
        rgb_img.transform((base_w, base_h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=0),
        dtype=np.uint8,
    )
    transformed_alpha = np.array(
        alpha_img.transform((base_w, base_h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR, fillcolor=0),
        dtype=np.uint8,
    )
    return transformed_img, transformed_alpha


def contrast_score(bg_frame: np.ndarray, image: np.ndarray, alpha: np.ndarray, x: int, y: int) -> float | None:
    h, w = alpha.shape
    patch = bg_frame[y : y + h, x : x + w].astype(np.float32)
    mask = alpha >= cfg.min_fg_alpha
    if mask.sum() == 0:
        return None
    fg = image.astype(np.float32)[mask]
    bg = patch[mask]
    rgb_diff = np.abs(fg.mean(axis=0) - bg.mean(axis=0)).mean()
    luma_diff = float(abs(luma(fg).mean() - luma(bg).mean()))
    if rgb_diff < cfg.min_contrast_rgb or luma_diff < cfg.min_contrast_luma:
        return None
    return rgb_diff + luma_diff


def sample_motion_step(rng: random.Random) -> tuple[float, float]:
    speed = rng.uniform(*cfg.moving_speed_frac_range)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return math.cos(theta) * speed, math.sin(theta) * speed


def frame_variants_for_proposals(overlay: PreparedOverlay) -> list[dict | None]:
    variants: list[dict | None] = []
    for frame_idx in range(cfg.SEQUENCE_LENGTH):
        rgb, alpha = apply_temporal_transform(
            overlay.proposal_frame_images[frame_idx],
            overlay.proposal_frame_alphas[frame_idx],
            overlay.geometry_mode,
            overlay.geometry_params,
            frame_idx,
        )
        bbox = alpha_bbox(alpha)
        if bbox is None:
            variants.append(None)
            continue
        bx0, by0, bx1, by1 = bbox
        cropped_alpha = alpha[by0:by1, bx0:bx1]
        visible_mask = (cropped_alpha.astype(np.float32) / 255.0) >= cfg.mask_alpha_threshold
        variants.append(
            {
                "bbox": bbox,
                "mask": visible_mask,
                "mask_area": int(visible_mask.sum()),
                "rgb": rgb,
                "alpha": alpha,
            }
        )
    return variants


def generate_proposals(frame: np.ndarray, overlay: PreparedOverlay, rng: random.Random) -> list[Proposal]:
    proposals: list[Proposal] = []
    frame_h, frame_w = frame.shape[:2]
    frame_variants = frame_variants_for_proposals(overlay)
    step_x = overlay.motion_step[0] * frame_w
    step_y = overlay.motion_step[1] * frame_w

    for _ in range(cfg.placement_trials_per_overlay):
        positions = []
        bboxes = []
        crop_boxes = []
        masks = []
        mask_areas = []
        valid = True
        anchor_initialized = False
        anchor_x0 = 0
        anchor_y0 = 0
        for frame_idx in range(cfg.SEQUENCE_LENGTH):
            variant = frame_variants[frame_idx]
            if variant is None:
                positions.append((0, 0))
                bboxes.append((0, 0, 0, 0))
                crop_boxes.append((0, 0, 0, 0))
                masks.append(np.zeros((0, 0), dtype=bool))
                mask_areas.append(0)
                continue
            bx0, by0, bx1, by1 = variant["bbox"]
            bw = bx1 - bx0
            bh = by1 - by0
            if bw <= 0 or bh <= 0:
                valid = False
                break
            if not anchor_initialized:
                if frame_w <= bw or frame_h <= bh:
                    valid = False
                    break
                anchor_x0 = rng.randint(0, frame_w - bw)
                anchor_y0 = rng.randint(0, frame_h - bh)
                anchor_initialized = True
            x = int(round(anchor_x0 + step_x * frame_idx))
            y = int(round(anchor_y0 + step_y * frame_idx))
            if x < 0 or y < 0 or x + bw > frame_w or y + bh > frame_h:
                valid = False
                break
            positions.append((x, y))
            bboxes.append((x, y, x + bw, y + bh))
            crop_boxes.append((bx0, by0, bx1, by1))
            masks.append(variant["mask"])
            mask_areas.append(variant["mask_area"])
        if not valid:
            continue
        first_visible_idx = next((idx for idx, item in enumerate(frame_variants) if item is not None), None)
        if first_visible_idx is None:
            continue
        first_variant = frame_variants[first_visible_idx]
        bx0, by0, bx1, by1 = crop_boxes[first_visible_idx]
        score = contrast_score(
            frame,
            first_variant["rgb"][by0:by1, bx0:bx1],
            first_variant["alpha"][by0:by1, bx0:bx1],
            bboxes[first_visible_idx][0],
            bboxes[first_visible_idx][1],
        )
        if score is None:
            continue
        proposals.append(Proposal(score=score, positions=positions, bboxes=bboxes, crop_boxes=crop_boxes, masks=masks, mask_areas=mask_areas))
        if len(proposals) >= cfg.proposal_attempts:
            break
    proposals.sort(key=lambda item: item.score, reverse=True)
    return proposals


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int((x1 - x0) * (y1 - y0))


def bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def mask_overlap_ratio(box_a, mask_a, box_b, mask_b) -> float:
    if not bboxes_overlap(box_a, box_b):
        return 0.0
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    ax0 = x0 - box_a[0]
    ay0 = y0 - box_a[1]
    ax1 = ax0 + (x1 - x0)
    ay1 = ay0 + (y1 - y0)
    bx0 = x0 - box_b[0]
    by0 = y0 - box_b[1]
    bx1 = bx0 + (x1 - x0)
    by1 = by0 + (y1 - y0)
    region_a = mask_a[ay0:ay1, ax0:ax1]
    region_b = mask_b[by0:by1, bx0:bx1]
    if region_a.size == 0 or region_b.size == 0:
        return 0.0
    overlap = np.logical_and(region_a, region_b).sum()
    if overlap == 0:
        return 0.0
    denom = max(1, min(mask_a.sum(), mask_b.sum()))
    return float(overlap) / float(denom)


def proposals_conflict(a: Proposal, b: Proposal) -> bool:
    for box_a, area_a, mask_a, box_b, area_b, mask_b in zip(a.bboxes, a.mask_areas, a.masks, b.bboxes, b.mask_areas, b.masks):
        if area_a == 0 or area_b == 0:
            continue
        inter = intersection_area(box_a, box_b)
        if inter == 0:
            continue
        overlap_upper_bound = inter / max(1, min(area_a, area_b))
        if overlap_upper_bound <= cfg.max_pairwise_mask_overlap_ratio:
            continue
        if mask_overlap_ratio(box_a, mask_a, box_b, mask_b) > cfg.max_pairwise_mask_overlap_ratio:
            return True
    return False


def select_non_overlapping(overlays: list[PreparedOverlay]) -> list[tuple[PreparedOverlay, Proposal]]:
    ordered = sorted(overlays, key=lambda item: (-(item.proposals[0].score if item.proposals else -1.0), len(item.proposals)))
    chosen: list[tuple[PreparedOverlay, Proposal]] = []
    for overlay in ordered:
        for proposal in overlay.proposals:
            if any(proposals_conflict(proposal, existing) for _, existing in chosen):
                continue
            chosen.append((overlay, proposal))
            break
    return chosen


def composite(frame: np.ndarray, overlay_rgb: np.ndarray, overlay_alpha: np.ndarray, x: int, y: int, alpha_scale: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    out = frame.copy()
    h, w = overlay_alpha.shape
    if h <= 0 or w <= 0:
        return out, np.zeros((0, 0), dtype=np.uint8), (0, 0, 0, 0)
    frame_h, frame_w = out.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x0 >= x1 or y0 >= y1:
        return out, np.zeros((0, 0), dtype=np.uint8), (0, 0, 0, 0)
    src_x0 = x0 - x
    src_y0 = y0 - y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)
    overlay_rgb = overlay_rgb[src_y0:src_y1, src_x0:src_x1]
    overlay_alpha = overlay_alpha[src_y0:src_y1, src_x0:src_x1]
    patch = out[y0:y1, x0:x1].astype(np.float32)
    alpha = (overlay_alpha.astype(np.float32) / 255.0) * alpha_scale
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    blended = overlay_rgb.astype(np.float32) * alpha + patch * (1.0 - alpha)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    mask = (alpha[..., 0] >= cfg.mask_alpha_threshold).astype(np.uint8) * 255
    return out, mask, (x0, y0, x1, y1)


def apply_global_seq_aug(frames: list[np.ndarray]) -> list[np.ndarray]:
    replay = None
    augmented_frames = []
    for frame in frames:
        if replay is None:
            out = cfg.global_seq_aug(image=frame)
            replay = out["replay"]
            augmented_frames.append(out["image"])
        else:
            out = cfg.global_seq_aug.replay(replay, image=frame)
            augmented_frames.append(out["image"])
    return augmented_frames


def spaced_grayscale_values(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [255]
    return [int(round(100 + idx * 155 / (count - 1))) for idx in range(count)]


def affine_2x3(matrix: np.ndarray) -> list[list[float]]:
    return [[float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])], [float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])]]


def bbox_center(box: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = box
    return [float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)]


def estimate_hidden_position(proposal: Proposal, frame_idx: int) -> tuple[int, int]:
    visible_indices = [idx for idx, box in enumerate(proposal.bboxes) if box[2] > box[0] and box[3] > box[1]]
    if not visible_indices:
        return (0, 0)
    if frame_idx in visible_indices:
        return proposal.positions[frame_idx]
    if len(visible_indices) == 1:
        return proposal.positions[visible_indices[0]]
    visible_indices = sorted(visible_indices)
    if frame_idx < visible_indices[0]:
        i0, i1 = visible_indices[0], visible_indices[1]
    elif frame_idx > visible_indices[-1]:
        i0, i1 = visible_indices[-2], visible_indices[-1]
    else:
        i0 = max(idx for idx in visible_indices if idx < frame_idx)
        i1 = min(idx for idx in visible_indices if idx > frame_idx)
    x0, y0 = proposal.positions[i0]
    x1, y1 = proposal.positions[i1]
    t = (frame_idx - i0) / max(i1 - i0, 1)
    return int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))


def frame_geometry_state(overlay: PreparedOverlay, proposal: Proposal, frame_idx: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    forward = temporal_forward_matrix(
        overlay.geometry_mode,
        overlay.geometry_params,
        frame_idx,
        overlay.proposal_frame_alphas[frame_idx],
    )
    box = proposal.bboxes[frame_idx]
    crop_box = proposal.crop_boxes[frame_idx]
    if box[2] > box[0] and box[3] > box[1] and crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        x, y = proposal.positions[frame_idx]
        bx0, by0, _, _ = crop_box
        placement = np.array([[1.0, 0.0, float(x - bx0)], [0.0, 1.0, float(y - by0)], [0.0, 0.0, 1.0]], dtype=np.float32)
        return placement @ forward, box
    visible_indices = [idx for idx, ref_box in enumerate(proposal.bboxes) if ref_box[2] > ref_box[0] and ref_box[3] > ref_box[1]]
    if not visible_indices:
        return np.eye(3, dtype=np.float32), (0, 0, 0, 0)
    ref_idx = min(visible_indices, key=lambda idx: abs(idx - frame_idx))
    ref_box = proposal.bboxes[ref_idx]
    ref_crop = proposal.crop_boxes[ref_idx]
    bw = ref_box[2] - ref_box[0]
    bh = ref_box[3] - ref_box[1]
    x, y = estimate_hidden_position(proposal, frame_idx)
    bx0, by0, _, _ = ref_crop
    placement = np.array([[1.0, 0.0, float(x - bx0)], [0.0, 1.0, float(y - by0)], [0.0, 0.0, 1.0]], dtype=np.float32)
    return placement @ forward, (x, y, x + bw, y + bh)


def overlay_export_record(overlay: PreparedOverlay, proposal: Proposal, mask_paths: list[str]) -> dict:
    mid_idx = len(mask_paths) // 2
    last_idx = len(mask_paths) - 1
    world_0, box0 = frame_geometry_state(overlay, proposal, 0)
    world_mid, box_mid = frame_geometry_state(overlay, proposal, mid_idx)
    world_last, box_last = frame_geometry_state(overlay, proposal, last_idx)
    affine_0_to_mid = world_mid @ np.linalg.inv(world_0)
    affine_0_to_last = world_last @ np.linalg.inv(world_0)
    affine_2_to_mid = world_mid @ np.linalg.inv(world_last)
    alpha0 = float(overlay.alpha_scales[0])
    alpha1 = float(overlay.alpha_scales[mid_idx])
    alpha2 = float(overlay.alpha_scales[last_idx])
    return {
        "object_id": f"{overlay.overlay_id:03d}",
        "mask_paths": mask_paths,
        "alpha": 1.0,
        "beta": 0.0,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "mode": overlay.mode,
        "A": affine_2x3(affine_0_to_mid),
        "B": affine_2x3(affine_2_to_mid),
        "geometry": {
            "center0": bbox_center(box0),
            "center1": bbox_center(box_mid),
            "center2": bbox_center(box_last),
            "bbox0": list(map(int, box0)),
            "bbox1": list(map(int, box_mid)),
            "bbox2": list(map(int, box_last)),
            "affine_0_to_2": affine_2x3(affine_0_to_last),
        },
    }


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_sequence_items(source_root: Path) -> list[SequenceItem]:
    items: list[SequenceItem] = []
    for video_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        seq_dirs = sorted(path for path in video_dir.iterdir() if path.is_dir())
        if not seq_dirs:
            continue
        chosen = seq_dirs[:1] if cfg.FIRST_TRIPLET_ONLY else seq_dirs
        for seq_dir in chosen:
            items.append(SequenceItem(video_name=video_dir.name, seq_name=seq_dir.name, path=seq_dir))
    return items


def find_frame_paths(seq_dir: Path) -> list[Path]:
    preferred = []
    for stem in ["I0", "I1", "I2"]:
        matches = [p for p in seq_dir.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTS]
        if len(matches) == 1:
            preferred.append(matches[0])
    if len(preferred) == 3:
        return preferred
    raw = []
    for stem in ["00000", "00001", "00002"]:
        matches = [p for p in seq_dir.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTS]
        if len(matches) == 1:
            raw.append(matches[0])
    if len(raw) == 3:
        return raw
    return sorted(
        p for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_bg" not in p.stem and not p.stem.startswith("M")
    )[:3]


def load_sequence_frames(seq_dir: Path) -> tuple[list[np.ndarray], str]:
    frame_paths = find_frame_paths(seq_dir)
    if len(frame_paths) != 3:
        raise RuntimeError(f"Expected 3 frames in {seq_dir}, found {len(frame_paths)}")
    return [load_rgb(path) for path in frame_paths], frame_paths[0].suffix.lower()


def copy_optional_support_files(source_seq_dir: Path, out_dir: Path) -> None:
    optional_names = ["I0_bg.png", "I1_bg.png", "I2_bg.png", "I1_bg_inter.png"]
    for name in optional_names:
        src = source_seq_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def make_derangement(size: int, rng: random.Random) -> list[int]:
    if size < 2:
        return list(range(size))
    indices = list(range(size))
    while True:
        perm = indices[:]
        rng.shuffle(perm)
        if all(i != perm[i] for i in indices):
            return perm


def apply_dark_overlay(frame: np.ndarray, alpha: float) -> np.ndarray:
    dark = np.clip(frame.astype(np.float32) * (1.0 - alpha), 0, 255)
    return dark.astype(np.uint8)


def sample_adjacent_overlay(asset: OverlayAsset, catalog: dict[str, list[CatalogEntry]], rng: random.Random) -> OverlayAsset:
    subfolder = asset.meta["subfolder"]
    entry = next(entry for entry in catalog[asset.cls_name] if entry.subfolder == subfolder)
    source_path = cfg.GOONS_ROOT / asset.source_rel
    idx = entry.files.index(source_path)
    neighbors = []
    if idx > 0:
        neighbors.append(entry.files[idx - 1])
    if idx + 1 < len(entry.files):
        neighbors.append(entry.files[idx + 1])
    if not neighbors:
        return asset
    chosen = rng.choice(neighbors)
    if asset.cls_name == "Font":
        image, alpha, font_meta = render_font_asset(
            chosen,
            rng,
            canvas_size=(asset.image.shape[1], asset.image.shape[0]),
            text=asset.meta.get("text"),
            font_size=asset.meta.get("font_size"),
            fill=tuple(asset.meta.get("fill", cfg.font_fill_palette[0])),
            stroke_width=asset.meta.get("stroke_width"),
        )
        return OverlayAsset(asset.cls_name, str(chosen.relative_to(cfg.GOONS_ROOT)), image, alpha, {"subfolder": subfolder, **font_meta})
    image, alpha = load_rgba_asset(chosen)
    return OverlayAsset(asset.cls_name, str(chosen.relative_to(cfg.GOONS_ROOT)), image, alpha, {"subfolder": subfolder})


def build_text_change_variants(asset: OverlayAsset, image: np.ndarray, alpha: np.ndarray, aug_replay: dict, new_size: tuple[int, int], font_paths: list[Path], rng: random.Random) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    if asset.cls_name == "Font":
        font_path = cfg.GOONS_ROOT / asset.source_rel
        base_text = asset.meta.get("text", "")
        base_fill = tuple(asset.meta.get("fill", cfg.font_fill_palette[0]))
        stroke_width = asset.meta.get("stroke_width")
        font_size = asset.meta.get("font_size")
        transition_mode = sample_text_transition_mode(rng)
        if transition_mode == "none_to_text":
            raw_base_image = np.zeros_like(asset.image)
            raw_base_alpha = np.zeros_like(asset.alpha)
            alt_text = random_text(rng, length=len(base_text) if base_text else None)
        elif transition_mode == "text_to_none":
            raw_base_image, raw_base_alpha = asset.image.copy(), asset.alpha.copy()
            alt_text = ""
        else:
            raw_base_image, raw_base_alpha = asset.image.copy(), asset.alpha.copy()
            alt_text = random_text_same_length(base_text, rng)
        if transition_mode == "text_to_none":
            raw_alt_image = np.zeros_like(raw_base_image)
            raw_alt_alpha = np.zeros_like(raw_base_alpha)
        else:
            raw_alt_image, raw_alt_alpha, _ = render_font_asset(
                font_path,
                rng,
                canvas_size=(asset.image.shape[1], asset.image.shape[0]),
                text=alt_text,
                font_size=font_size,
                fill=base_fill,
                stroke_width=stroke_width,
            )
        base_image, base_alpha, _ = apply_overlay_aug(raw_base_image, raw_base_alpha, replay=aug_replay)
        alt_image, alt_alpha, _ = apply_overlay_aug(raw_alt_image, raw_alt_alpha, replay=aug_replay)
        base_image, base_alpha = resize_overlay(base_image, base_alpha, new_size)
        alt_image, alt_alpha = resize_overlay(alt_image, alt_alpha, new_size)
        return [base_image, base_image, alt_image], [base_alpha, base_alpha, alt_alpha], "texture_change"
    font_path = rng.choice(font_paths)
    transition_mode = sample_text_transition_mode(rng)
    base_text = random_text(rng)
    alt_text = random_text_same_length(base_text, rng)
    fill = (255, 255, 255) if rng.random() < cfg.p_white_text else rng.choice(cfg.font_fill_palette)
    stroke_width = rng.randint(*cfg.font_stroke_width)
    base_image = image.copy()
    alt_image = image.copy()
    if transition_mode == "none_to_text":
        base_image = image.copy()
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)
    elif transition_mode == "text_to_none":
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
    else:
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)
    return [base_image, base_image, alt_image], [alpha.copy(), alpha.copy(), alpha.copy()], "texture_change"


def half_overlay(image: np.ndarray, alpha: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    out_image = image.copy()
    out_alpha = alpha.copy()
    if rng.random() < 0.5:
        split = w // 2
        if rng.random() < 0.5:
            out_image[:, :split] = 0
            out_alpha[:, :split] = 0
            # Keep the original bbox extent without visible content so placement does not re-anchor.
            out_alpha[:, 0] = 1
        else:
            out_image[:, split:] = 0
            out_alpha[:, split:] = 0
            out_alpha[:, -1] = 1
    else:
        split = h // 2
        if rng.random() < 0.5:
            out_image[:split, :] = 0
            out_alpha[:split, :] = 0
            out_alpha[0, :] = 1
        else:
            out_image[split:, :] = 0
            out_alpha[split:, :] = 0
            out_alpha[-1, :] = 1
    return out_image, out_alpha


def maybe_common_object_alpha(dataset_key: str, rng: random.Random, existing_dynamic: bool = False) -> list[float]:
    if existing_dynamic:
        return [1.0, 1.0, 1.0]
    if dataset_key.startswith("object_") and rng.random() < 0.2:
        return [cfg.transparent_alpha] * cfg.SEQUENCE_LENGTH
    return [1.0] * cfg.SEQUENCE_LENGTH


def prepare_overlay(asset: OverlayAsset, frame_size: tuple[int, int], catalog: dict[str, list[CatalogEntry]], font_paths: list[Path], rng: random.Random, overlay_id: int) -> PreparedOverlay:
    frame_w, frame_h = frame_size
    image, alpha, aug_replay = apply_overlay_aug(asset.image, asset.alpha)
    frame_area = float(frame_w * frame_h)
    overlay_area = float(image.shape[0] * image.shape[1])
    target_area_ratio = rng.uniform(cfg.min_area_ratio, cfg.max_area_ratio)
    target_overlay_area = target_area_ratio * frame_area
    scale = math.sqrt(target_overlay_area / max(overlay_area, 1.0))
    new_w = max(8, int(round(image.shape[1] * scale)))
    new_h = max(8, int(round(image.shape[0] * scale)))
    image, alpha = resize_overlay(image, alpha, (new_w, new_h))

    key = cfg.DATASET_KEY
    frame_images = [image.copy() for _ in range(cfg.SEQUENCE_LENGTH)]
    frame_alphas = [alpha.copy() for _ in range(cfg.SEQUENCE_LENGTH)]
    proposal_frame_images = [image.copy() for _ in range(cfg.SEQUENCE_LENGTH)]
    proposal_frame_alphas = [alpha.copy() for _ in range(cfg.SEQUENCE_LENGTH)]
    alpha_scales = [1.0] * cfg.SEQUENCE_LENGTH
    motion_step = (0.0, 0.0)
    geometry_mode = "static"
    geometry_params: dict = {}
    mode = "static"

    if key.startswith("consistent_"):
        if "transparent" in key:
            alpha_scales = [rng.uniform(*cfg.transparent_alpha_range)] * cfg.SEQUENCE_LENGTH
        if "motion" in key:
            motion_step = sample_motion_step(rng)
            if rng.random() < 0.3:
                geometry_mode = "affine"
                geometry_params = sample_affine_params(rng)
        mode = "static"
    elif key == "scenechange_preserved_go":
        if rng.random() < 0.3:
            motion_step = sample_motion_step(rng)
        if rng.random() < 0.3:
            alpha_scales = [cfg.transparent_alpha] * cfg.SEQUENCE_LENGTH
        mode = "static"
    elif key == "object_disappear":
        motion_step = sample_motion_step(rng) if rng.random() < 0.2 else (0.0, 0.0)
        alpha_scales = maybe_common_object_alpha(key, rng)
        if rng.random() < 0.5:
            frame_alphas[2] = np.zeros_like(alpha)
            frame_images[2] = np.zeros_like(image)
            mode = "disappear"
        else:
            frame_alphas[0] = np.zeros_like(alpha)
            frame_alphas[1] = np.zeros_like(alpha)
            frame_images[0] = np.zeros_like(image)
            frame_images[1] = np.zeros_like(image)
            mode = "appear"
    elif key == "object_texture_change":
        motion_step = sample_motion_step(rng) if rng.random() < 0.2 else (0.0, 0.0)
        alpha_scales = maybe_common_object_alpha(key, rng)
        frame_images, frame_alphas, mode = build_text_change_variants(asset, image, alpha, aug_replay, (new_w, new_h), font_paths, rng)
    elif key == "object_color_change":
        motion_step = sample_motion_step(rng) if rng.random() < 0.2 else (0.0, 0.0)
        alpha_scales = maybe_common_object_alpha(key, rng)
        alt_image, _ = apply_color_change(image.copy(), rng)
        frame_images = [image.copy(), image.copy(), alt_image]
        mode = "color_change"
    elif key == "object_visibility_change":
        motion_step = sample_motion_step(rng) if rng.random() < 0.2 else (0.0, 0.0)
        if rng.random() < 0.3:
            end_image = apply_color_scale_only(image.copy(), rng, *cfg.visibility_change_color_scale_range)
            frame_images = [
                image.copy(),
                np.clip(0.5 * image.astype(np.float32) + 0.5 * end_image.astype(np.float32), 0, 255).astype(np.uint8),
                end_image,
            ]
            alpha_scales = maybe_common_object_alpha(key, rng)
        else:
            start_alpha = rng.uniform(0.6, 1.0)
            end_alpha = rng.uniform(0.6, 1.0)
            alpha_scales = [start_alpha, 0.5 * (start_alpha + end_alpha), end_alpha]
        mode = "visibility_change"
    elif key == "object_shape_change":
        alpha_scales = maybe_common_object_alpha(key, rng)
        draw = rng.random()
        if draw < 0.4:
            cut_image, cut_alpha = half_overlay(image, alpha, rng)
            proposal_frame_images = [image.copy(), image.copy(), image.copy()]
            proposal_frame_alphas = [alpha.copy(), alpha.copy(), alpha.copy()]
            if rng.random() < 0.5:
                frame_images = [image.copy(), image.copy(), cut_image]
                frame_alphas = [alpha.copy(), alpha.copy(), cut_alpha]
            else:
                frame_images = [cut_image, cut_image, image.copy()]
                frame_alphas = [cut_alpha, cut_alpha, alpha.copy()]
        elif draw < 0.65:
            if rng.random() < 0.5:
                alt_image = np.ascontiguousarray(np.flip(image, axis=1))
                alt_alpha = np.ascontiguousarray(np.flip(alpha, axis=1))
            else:
                alt_image = np.ascontiguousarray(np.flip(image, axis=0))
                alt_alpha = np.ascontiguousarray(np.flip(alpha, axis=0))
            frame_images = [image.copy(), image.copy(), alt_image]
            frame_alphas = [alpha.copy(), alpha.copy(), alt_alpha]
        else:
            adjacent = sample_adjacent_overlay(asset, catalog, rng)
            alt_image, alt_alpha, _ = apply_overlay_aug(adjacent.image, adjacent.alpha, replay=aug_replay)
            alt_image, alt_alpha = resize_overlay(alt_image, alt_alpha, (new_w, new_h))
            frame_images = [image.copy(), image.copy(), alt_image]
            frame_alphas = [alpha.copy(), alpha.copy(), alt_alpha]
        mode = "shape_change"
    else:
        raise RuntimeError(f"Unsupported dataset key: {key}")

    canvas_params = geometry_params if geometry_mode == "affine" else {}
    if geometry_mode == "affine":
        canvas_w, canvas_h = affine_canvas_size(frame_images[0].shape[1], frame_images[0].shape[0], canvas_params)
    else:
        canvas_w, canvas_h = frame_images[0].shape[1], frame_images[0].shape[0]
    centered_images: list[np.ndarray] = []
    centered_alphas: list[np.ndarray] = []
    for img, alp in zip(frame_images, frame_alphas):
        centered_img, centered_alpha = center_on_canvas(img, alp, (canvas_w, canvas_h))
        centered_images.append(centered_img)
        centered_alphas.append(centered_alpha)
    frame_images = centered_images
    frame_alphas = centered_alphas
    centered_proposal_images: list[np.ndarray] = []
    centered_proposal_alphas: list[np.ndarray] = []
    for img, alp in zip(proposal_frame_images, proposal_frame_alphas):
        centered_img, centered_alpha = center_on_canvas(img, alp, (canvas_w, canvas_h))
        centered_proposal_images.append(centered_img)
        centered_proposal_alphas.append(centered_alpha)
    proposal_frame_images = centered_proposal_images
    proposal_frame_alphas = centered_proposal_alphas

    return PreparedOverlay(
        overlay_id=overlay_id,
        asset=asset,
        frame_images=frame_images,
        frame_alphas=frame_alphas,
        proposal_frame_images=proposal_frame_images,
        proposal_frame_alphas=proposal_frame_alphas,
        mode=mode,
        geometry_mode=geometry_mode,
        geometry_params=geometry_params,
        motion_step=motion_step,
        alpha_scales=alpha_scales,
        color=0,
        proposals=[],
    )


def build_augmented_sequence(seq_dir: Path, out_dir: Path, catalog: dict[str, list[CatalogEntry]], font_paths: list[Path], rng: random.Random) -> dict:
    frames, frame_ext = load_sequence_frames(seq_dir)
    ensure_empty_dir(out_dir)
    overlay_masks_root = out_dir / "overlays_masks"
    aggregate_root = out_dir / "aggregate_masks"
    overlay_masks_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)
    copy_optional_support_files(seq_dir, out_dir)

    chosen: list[tuple[PreparedOverlay, Proposal]] | None = None
    for _ in range(cfg.SEQUENCE_BUILD_ATTEMPTS):
        prepared: list[PreparedOverlay] = []
        for overlay_id, cls_name in enumerate(cfg.CLASS_ORDER):
            asset = sample_overlay_asset_for_class(catalog, cls_name, rng)
            overlay = prepare_overlay(asset, (frames[0].shape[1], frames[0].shape[0]), catalog, font_paths, rng, overlay_id)
            overlay.proposals = generate_proposals(frames[0], overlay, rng)
            if not overlay.proposals:
                prepared = []
                break
            prepared.append(overlay)
        if len(prepared) != len(cfg.CLASS_ORDER):
            continue
        chosen = select_non_overlapping(prepared)
        if len(chosen) == len(cfg.CLASS_ORDER):
            break
    if chosen is None or len(chosen) != len(cfg.CLASS_ORDER):
        raise RuntimeError(f"Failed to place all five GOoNS classes exactly once for {seq_dir}")

    overlays_meta = []
    aggregate_masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
    out_frames = [frame.copy() for frame in frames]
    grayscale_values = spaced_grayscale_values(len(chosen))
    rng.shuffle(grayscale_values)
    for overlay, proposal in chosen:
        overlay.color = grayscale_values.pop()
        mask_paths: list[str] = []
        for frame_idx, base_frame in enumerate(out_frames):
            variant_rgb, variant_alpha = apply_temporal_transform(
                overlay.frame_images[frame_idx],
                overlay.frame_alphas[frame_idx],
                overlay.geometry_mode,
                overlay.geometry_params,
                frame_idx,
            )
            bx0, by0, bx1, by1 = proposal.crop_boxes[frame_idx]
            variant_rgb = variant_rgb[by0:by1, bx0:bx1]
            variant_alpha = variant_alpha[by0:by1, bx0:bx1]
            x, y = proposal.positions[frame_idx]
            out_frames[frame_idx], mask, clipped_box = composite(base_frame, variant_rgb, variant_alpha, x, y, overlay.alpha_scales[frame_idx])
            mask_name = f"{overlay.overlay_id:03d}_M{frame_idx}.png"
            mask_path = overlay_masks_root / mask_name
            full_mask = np.zeros(base_frame.shape[:2], dtype=np.uint8)
            if mask.size > 0:
                x0, y0, x1, y1 = clipped_box
                full_mask[y0:y1, x0:x1] = mask
                patch = aggregate_masks[frame_idx][y0:y1, x0:x1]
                patch[mask > 0] = overlay.color
            save_gray(mask_path, full_mask)
            mask_paths.append(mask_name)
        overlays_meta.append(overlay_export_record(overlay, proposal, mask_paths))
    out_frames = apply_global_seq_aug(out_frames)

    for idx, image in enumerate(out_frames):
        save_rgb(out_dir / f"I{idx}{frame_ext}", image)
        save_gray(aggregate_root / f"M{idx}.png", aggregate_masks[idx])
    (out_dir / "overlays.json").write_text(json.dumps(overlays_meta, indent=2))
    return {"sequence": str(seq_dir), "augmented": True, "overlay_count": len(overlays_meta)}


def build_scenechange_sequence(item: SequenceItem, perm_item: SequenceItem, out_dir: Path, rng: random.Random) -> dict:
    frames_a, frame_ext = load_sequence_frames(item.path)
    frames_b, _ = load_sequence_frames(perm_item.path)
    if rng.random() < 0.4:
        frames = [frames_a[0], frames_a[1], frames_b[2]]
        scenechange_mode = "permutation"
        dark_alpha = None
    else:
        dark_alpha = rng.uniform(0.7, 0.9)
        if rng.random() < 0.5:
            frames = [frames_a[0], frames_a[1], apply_dark_overlay(frames_a[2], dark_alpha)]
            scenechange_mode = "darken_third"
        else:
            frames = [
                apply_dark_overlay(frames_a[0], dark_alpha),
                apply_dark_overlay(frames_a[1], dark_alpha),
                frames_a[2],
            ]
            scenechange_mode = "darken_first_second"
    ensure_empty_dir(out_dir)
    (out_dir / "overlays_masks").mkdir(parents=True, exist_ok=True)
    aggregate_root = out_dir / "aggregate_masks"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        save_rgb(out_dir / f"I{idx}{frame_ext}", frame)
        save_gray(aggregate_root / f"M{idx}.png", np.zeros(frame.shape[:2], dtype=np.uint8))
    (out_dir / "overlays.json").write_text("[]\n")
    return {
        "sequence": f"{item.video_name}/{item.seq_name}",
        "augmented": False,
        "scenechange_mode": scenechange_mode,
        "dark_alpha": dark_alpha,
        "scenechange_source": perm_item.video_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one DAVIS triplet mini-project dataset from a config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a config .py file under davis_triplet_miniproject/configs.")
    args = parser.parse_args()

    global cfg
    cfg = load_config(args.config.resolve())
    rng = random.Random(cfg.SEED)
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not cfg.SOURCE_ROOT.exists():
        raise SystemExit(f"Source root does not exist: {cfg.SOURCE_ROOT}")

    if cfg.DATASET_KEY == "scenechange":
        items = list_sequence_items(cfg.SOURCE_ROOT)
        perm = make_derangement(len(items), rng)
        for idx, item in enumerate(items):
            out_dir = cfg.OUTPUT_ROOT / item.video_name / item.seq_name
            build_scenechange_sequence(item, items[perm[idx]], out_dir, rng)
        print(f"Built {len(items)} scenechange sequences into {cfg.OUTPUT_ROOT}")
        return

    catalog = load_catalog()
    font_paths = sample_font_paths(catalog)
    items = list_sequence_items(cfg.SOURCE_ROOT)
    built = 0
    for item in items:
        out_dir = cfg.OUTPUT_ROOT / item.video_name / item.seq_name
        build_augmented_sequence(item.path, out_dir, catalog, font_paths, rng)
        built += 1
    print(f"Built {built} sequences into {cfg.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
