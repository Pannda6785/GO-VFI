#!/usr/bin/env python3
"""Augment DAVIS clip datasets with GOoNS overlays."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as cfg


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
    image: np.ndarray
    alpha: np.ndarray
    alt_image: np.ndarray | None
    alt_alpha: np.ndarray | None
    temporal_mode: str
    appearance_change_alt_first: bool
    temporal_params: dict
    alpha_scale: float
    color: int
    proposals: list[Proposal]


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

    # Fallback for assets that converted to RGBA without transparency.
    corners = np.stack(
        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0
    ).astype(np.float32)
    bg = corners.mean(axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    alpha = np.where(dist > 18.0, 255, 0).astype(np.uint8)
    if alpha.max() == 0:
        alpha.fill(255)
    return rgb, alpha


def resize_overlay(image: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    rgb_img = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
    alpha_img = Image.fromarray(alpha, mode="L").resize((width, height), Image.Resampling.LANCZOS)
    return np.array(rgb_img, dtype=np.uint8), np.array(alpha_img, dtype=np.uint8)


def center_on_canvas(
    image: np.ndarray,
    alpha: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    canvas_w, canvas_h = canvas_size
    src_h, src_w = image.shape[:2]
    out_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    out_alpha = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # If the source is larger than the target canvas, crop it symmetrically first.
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


def temporal_canvas_size(mode: str, width: int, height: int, params: dict | None = None) -> tuple[int, int]:
    if mode == "affine":
        return affine_canvas_size(width, height, params or {})
    return width, height


def sample_temporal_params(mode: str, rng: random.Random) -> dict:
    if mode == "affine":
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
    return {}


def apply_temporal_transform(
    image: np.ndarray,
    alpha: np.ndarray,
    mode: str,
    params: dict,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        rgb_img.transform(
            (base_w, base_h),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BICUBIC,
            fillcolor=0,
        ),
        dtype=np.uint8,
    )
    transformed_alpha = np.array(
        alpha_img.transform(
            (base_w, base_h),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        ),
        dtype=np.uint8,
    )
    return transformed_img, transformed_alpha


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
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale = np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shear_m = np.array(
        [
            [1.0, shear, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return back @ rot @ shear_m @ scale @ to_origin


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
        box_w = bbox[2] - bbox[0] + padding * 2
        box_h = bbox[3] - bbox[1] + padding * 2
        if box_w <= target_w and box_h <= target_h:
            break
        size -= 2

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    x = max(0, (target_w - (bbox[2] - bbox[0])) // 2 - bbox[0])
    y = max(0, (target_h - (bbox[3] - bbox[1])) // 2 - bbox[1])
    draw.text(
        (x, y),
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

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


def blank_overlay_like(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros_like(image), np.zeros_like(alpha)


def random_text_same_length(original: str, rng: random.Random) -> str:
    generated = "".join(
        rng.choice(cfg.font_text_alphabet) if ch != " " else " "
        for ch in original
    )
    if generated == original:
        generated = generated[::-1]
    return generated


def random_text(rng: random.Random, length: int | None = None) -> str:
    length = length if length is not None else rng.randint(*cfg.font_text_length_range)
    return "".join(rng.choice(cfg.font_text_alphabet) for _ in range(length))


def sample_text_transition_mode(rng: random.Random) -> str:
    modes = list(cfg.appearance_change_text_transition_probs.keys())
    weights = list(cfg.appearance_change_text_transition_probs.values())
    return weighted_choice(modes, weights, rng)


def apply_color_change(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    lo, hi = cfg.appearance_change_color_scale_range
    factors = np.array(
        [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)],
        dtype=np.float32,
    )
    perm = np.array(rng.sample([0, 1, 2], 3))
    out = np.clip(image.astype(np.float32) * factors[None, None, :], 0, 255).astype(np.uint8)
    out = out[..., perm]
    return out, {
        "color_multipliers": factors.tolist(),
        "color_permutation": perm.tolist(),
    }


def sample_font_paths(catalog: dict[str, list[CatalogEntry]]) -> list[Path]:
    paths: list[Path] = []
    for entry in catalog.get("Font", []):
        paths.extend(entry.files)
    return sorted(set(paths))


def alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_alpha_bbox(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbox = alpha_bbox(alpha)
    if bbox is None:
        return image, alpha
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


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
    while size >= 10 and (
        (bbox_text[2] - bbox_text[0]) > box_w or (bbox_text[3] - bbox_text[1]) > box_h
    ):
        size -= 2
        font = ImageFont.truetype(str(font_path), size=size)
        bbox_text = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)

    text_x = x0 + max(0, (box_w - (bbox_text[2] - bbox_text[0])) // 2 - bbox_text[0])
    text_y = y0 + max(0, (box_h - (bbox_text[3] - bbox_text[1])) // 2 - bbox_text[1])
    draw.text(
        (text_x, text_y),
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    rendered = np.array(overlay, dtype=np.uint8)
    out = base_image.copy()
    region = alpha > 0
    out[region] = rendered[region]
    return out


def build_font_appearance_change(
    asset: OverlayAsset,
    aug_replay: dict,
    new_size: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    font_path = cfg.GOONS_ROOT / asset.source_rel
    base_text = asset.meta.get("text", "")
    base_fill = tuple(asset.meta.get("fill", cfg.font_fill_palette[0]))
    stroke_width = asset.meta.get("stroke_width")
    font_size = asset.meta.get("font_size")
    transition_mode = sample_text_transition_mode(rng)
    change_color = rng.random() < cfg.appearance_change_text_and_color_prob
    alt_fill = base_fill
    if change_color:
        alt_fill = (255, 255, 255) if rng.random() < cfg.p_white_text else rng.choice(cfg.font_fill_palette)

    if transition_mode == "none_to_text":
        raw_base_image, raw_base_alpha = blank_overlay_like(asset.image, asset.alpha)
        alt_text = random_text(rng, length=len(base_text) if base_text else None)
    elif transition_mode == "text_to_none":
        raw_base_image, raw_base_alpha = asset.image.copy(), asset.alpha.copy()
        alt_text = ""
    else:
        raw_base_image, raw_base_alpha = asset.image.copy(), asset.alpha.copy()
        alt_text = random_text_same_length(base_text, rng)

    if transition_mode == "text_to_none":
        raw_alt_image, raw_alt_alpha = blank_overlay_like(raw_base_image, raw_base_alpha)
    else:
        raw_alt_image, raw_alt_alpha, _ = render_font_asset(
            font_path,
            rng,
            canvas_size=(asset.image.shape[1], asset.image.shape[0]),
            text=alt_text,
            font_size=font_size,
            fill=alt_fill,
            stroke_width=stroke_width,
        )

    base_image, base_alpha, _ = apply_overlay_aug(raw_base_image, raw_base_alpha, replay=aug_replay)
    alt_image, alt_alpha, _ = apply_overlay_aug(raw_alt_image, raw_alt_alpha, replay=aug_replay)
    base_image, base_alpha = resize_overlay(base_image, base_alpha, new_size)
    alt_image, alt_alpha = resize_overlay(alt_image, alt_alpha, new_size)
    meta = {
        "alt_variant": "font_text_change",
        "text_transition_mode": transition_mode,
        "text_before": base_text if transition_mode != "none_to_text" else "",
        "alt_text": alt_text,
    }
    if change_color:
        meta["text_color_change"] = True
        meta["fill_before"] = list(base_fill)
        meta["fill_after"] = list(alt_fill)
    return base_image, base_alpha, alt_image, alt_alpha, meta


def build_nonfont_text_appearance_change(
    image: np.ndarray,
    alpha: np.ndarray,
    font_path: Path,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, dict]:
    transition_mode = sample_text_transition_mode(rng)
    base_text = random_text(rng)
    alt_text = random_text_same_length(base_text, rng)
    fill = (255, 255, 255) if rng.random() < cfg.p_white_text else rng.choice(cfg.font_fill_palette)
    stroke_width = rng.randint(*cfg.font_stroke_width)
    change_color = rng.random() < cfg.appearance_change_text_and_color_prob

    base_image = image.copy()
    alt_image = image.copy()
    meta = {
        "alt_variant": "mask_bounded_text",
        "text_transition_mode": transition_mode,
        "text_font": str(font_path.relative_to(cfg.GOONS_ROOT)),
        "text_fill": list(fill),
        "text_stroke_width": stroke_width,
    }

    if transition_mode == "none_to_text":
        base_text = ""
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)
    elif transition_mode == "text_to_none":
        alt_text = ""
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
    else:
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)

    meta["text_before"] = base_text
    meta["alt_text"] = alt_text

    if change_color:
        alt_image, color_meta = apply_color_change(alt_image, rng)
        meta["text_color_change"] = True
        meta.update(color_meta)

    return base_image, alt_image, meta


def choose_overlay_count(rng: random.Random) -> int:
    range_pair = weighted_choice(cfg.count_ranges, cfg.count_ranges_prob, rng)
    lo, hi = range_pair
    return rng.randint(lo, hi)


def sample_class(catalog: dict[str, list[CatalogEntry]], rng: random.Random) -> str:
    classes = [cls_name for cls_name in cfg.overlay_class_prob if catalog.get(cls_name)]
    weights = [cfg.overlay_class_prob[cls_name] for cls_name in classes]
    return weighted_choice(classes, weights, rng)


def sample_catalog_entry(catalog: dict[str, list[CatalogEntry]], cls_name: str, rng: random.Random) -> CatalogEntry:
    entries = catalog[cls_name]
    return weighted_choice(entries, [entry.weight for entry in entries], rng)


def sample_blink_alphas(rng: random.Random) -> tuple[float, float]:
    lo, hi = cfg.blink_alpha_range
    a = rng.uniform(lo, hi)
    b = rng.uniform(lo, hi)
    if rng.random() < abs(a - b) ** cfg.blink_alpha_gap_bias_power / max((hi - lo) ** cfg.blink_alpha_gap_bias_power, 1e-8):
        return a, b

    anchor = rng.uniform(lo, hi)
    far_edge = hi if (anchor - lo) < (hi - anchor) else lo
    return anchor, far_edge


def sample_overlay_asset(catalog: dict[str, list[CatalogEntry]], rng: random.Random) -> OverlayAsset:
    cls_name = sample_class(catalog, rng)
    entry = sample_catalog_entry(catalog, cls_name, rng)
    source_path = rng.choice(entry.files)

    if cls_name == "Font":
        image, alpha, font_meta = render_font_asset(source_path, rng)
        source_rel = str(source_path.relative_to(cfg.GOONS_ROOT))
        return OverlayAsset(
            cls_name=cls_name,
            source_rel=source_rel,
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


def prepare_overlay(
    asset: OverlayAsset,
    frame_size: tuple[int, int],
    rng: random.Random,
    overlay_id: int,
    font_paths: list[Path],
) -> PreparedOverlay:
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

    temporal_mode = weighted_choice(
        list(cfg.temporal_mode_probs.keys()),
        list(cfg.temporal_mode_probs.values()),
        rng,
    )
    appearance_change_alt_first = rng.random() < 0.5
    temporal_params = sample_temporal_params(temporal_mode, rng)
    alpha_scale = cfg.transparent_alpha if rng.random() < cfg.p_transparent else 1.0
    alt_image = None
    alt_alpha = None

    if temporal_mode == "appearance_change":
        if asset.cls_name == "Font":
            image, alpha, alt_image, alt_alpha, text_meta = build_font_appearance_change(
                asset,
                aug_replay,
                (new_w, new_h),
                rng,
            )
            asset.meta.update(text_meta)
        else:
            alt_image = image.copy()
            alt_alpha = alpha.copy()
            if rng.random() < cfg.appearance_change_nonfont_color_only_prob:
                alt_image, color_meta = apply_color_change(alt_image, rng)
                asset.meta["alt_variant"] = "color_change"
                asset.meta.update(color_meta)
            else:
                font_path = rng.choice(font_paths)
                image, alt_image, text_meta = build_nonfont_text_appearance_change(
                    image,
                    alpha,
                    font_path,
                    rng,
                )
                asset.meta.update(text_meta)

    canvas_w, canvas_h = temporal_canvas_size(temporal_mode, image.shape[1], image.shape[0], temporal_params)
    image, alpha = center_on_canvas(image, alpha, (canvas_w, canvas_h))
    if alt_image is not None and alt_alpha is not None:
        alt_image, alt_alpha = center_on_canvas(alt_image, alt_alpha, (canvas_w, canvas_h))

    return PreparedOverlay(
        overlay_id=overlay_id,
        asset=asset,
        image=image,
        alpha=alpha,
        alt_image=alt_image,
        alt_alpha=alt_alpha,
        temporal_mode=temporal_mode,
        appearance_change_alt_first=appearance_change_alt_first,
        temporal_params=temporal_params,
        alpha_scale=alpha_scale,
        color=0,
        proposals=[],
    )


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def propose_movement(rng: random.Random) -> tuple[int, int]:
    if rng.random() >= cfg.p_moving:
        return 0, 0
    speed = rng.uniform(*cfg.moving_speed_frac_range)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    step = speed
    return (
        math.cos(theta) * step,
        math.sin(theta) * step,
    )


def middle_frame_index() -> int:
    return cfg.SEQUENCE_LENGTH // 2


def is_left_segment(frame_idx: int) -> bool:
    return frame_idx <= middle_frame_index()


def temporal_alpha_factor(mode: str, frame_idx: int, rng_seed: int, blink_alphas: tuple[float, float] | None = None) -> float:
    mid = middle_frame_index()
    if mode == "disappear":
        if cfg.drop_middle_frame_for_appear_disappear:
            return 1.0 if frame_idx < mid else 0.0
        return 1.0 if frame_idx <= mid else 0.0
    if mode == "appear":
        return 1.0 if frame_idx > mid else 0.0
    if mode == "blink":
        groups = [idx // 2 for idx in range(cfg.SEQUENCE_LENGTH)]
        start_half = (rng_seed % 2) == 0
        is_half = (groups[frame_idx] % 2 == 0) if start_half else (groups[frame_idx] % 2 == 1)
        low_alpha, high_alpha = blink_alphas if blink_alphas is not None else sample_blink_alphas(random.Random(rng_seed))
        return low_alpha if is_half else high_alpha
    return 1.0


def get_overlay_variant(overlay: PreparedOverlay, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if overlay.temporal_mode == "appearance_change" and overlay.alt_image is not None and overlay.alt_alpha is not None:
        use_alt = is_left_segment(frame_idx) if overlay.appearance_change_alt_first else not is_left_segment(frame_idx)
        if use_alt:
            return overlay.alt_image, overlay.alt_alpha
    return overlay.image, overlay.alpha


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


def generate_proposals(frame: np.ndarray, overlay: PreparedOverlay, rng: random.Random) -> list[Proposal]:
    proposals: list[Proposal] = []
    frame_h, frame_w = frame.shape[:2]
    frame_variants = []
    for frame_idx in range(cfg.SEQUENCE_LENGTH):
        variant_rgb, variant_alpha = get_overlay_variant(overlay, frame_idx)
        variant_rgb, variant_alpha = apply_temporal_transform(
            variant_rgb,
            variant_alpha,
            overlay.temporal_mode,
            overlay.temporal_params,
            frame_idx,
        )
        bbox = alpha_bbox(variant_alpha)
        if bbox is None:
            frame_variants.append(None)
            continue
        bx0, by0, bx1, by1 = bbox
        cropped_alpha = variant_alpha[by0:by1, bx0:bx1]
        visible_mask = (cropped_alpha.astype(np.float32) / 255.0) >= cfg.mask_alpha_threshold
        frame_variants.append(
            {
                "bbox": bbox,
                "mask": visible_mask,
                "mask_area": int(visible_mask.sum()),
                "rgb": variant_rgb,
                "alpha": variant_alpha,
            }
        )

    for _ in range(cfg.placement_trials_per_overlay):
        step_x_frac, step_y_frac = propose_movement(rng)
        step_x = step_x_frac * frame_w
        step_y = step_y_frac * frame_w
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
                anchor_x0 = rng.randint(0, frame_w - bw) - int(round(step_x * frame_idx))
                anchor_y0 = rng.randint(0, frame_h - bh) - int(round(step_y * frame_idx))
                anchor_initialized = True
            x = int(round(anchor_x0 + step_x * frame_idx))
            y = int(round(anchor_y0 + step_y * frame_idx))
            if x < 0 or y < 0 or x + bw > frame_w or y + bh > frame_h:
                valid = False
                break
            # `x, y` are already the intended top-left of the visible cropped box.
            # Keep those exact coordinates so the final cropped composite matches
            # the geometry used during proposal conflict checks.
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
        proposals.append(
            Proposal(
                score=score,
                positions=positions,
                bboxes=bboxes,
                crop_boxes=crop_boxes,
                masks=masks,
                mask_areas=mask_areas,
            )
        )
        if len(proposals) >= cfg.proposal_attempts:
            break

    proposals.sort(key=lambda item: item.score, reverse=True)
    return proposals


def bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int((x1 - x0) * (y1 - y0))


def mask_overlap_ratio(
    box_a: tuple[int, int, int, int],
    mask_a: np.ndarray,
    box_b: tuple[int, int, int, int],
    mask_b: np.ndarray,
) -> float:
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
    for box_a, area_a, mask_a, box_b, area_b, mask_b in zip(
        a.bboxes,
        a.mask_areas,
        a.masks,
        b.bboxes,
        b.mask_areas,
        b.masks,
    ):
        if area_a == 0 or area_b == 0:
            continue
        inter = intersection_area(box_a, box_b)
        if inter == 0:
            continue
        # Safe upper bound: true mask overlap cannot exceed bbox intersection area.
        overlap_upper_bound = inter / max(1, min(area_a, area_b))
        if overlap_upper_bound <= cfg.max_pairwise_mask_overlap_ratio:
            continue
        if mask_overlap_ratio(box_a, mask_a, box_b, mask_b) > cfg.max_pairwise_mask_overlap_ratio:
            return True
    return False


def select_non_overlapping(overlays: list[PreparedOverlay]) -> list[tuple[PreparedOverlay, Proposal]]:
    ordered = sorted(
        overlays,
        key=lambda item: (
            -(item.proposals[0].score if item.proposals else -1.0),
            len(item.proposals),
            -(item.alpha.shape[0] * item.alpha.shape[1]),
        ),
    )
    chosen: list[tuple[PreparedOverlay, Proposal]] = []
    for overlay in ordered:
        for proposal in overlay.proposals:
            if any(proposals_conflict(proposal, existing) for _, existing in chosen):
                continue
            chosen.append((overlay, proposal))
            break
    return chosen


def composite(
    frame: np.ndarray,
    overlay_rgb: np.ndarray,
    overlay_alpha: np.ndarray,
    x: int,
    y: int,
    alpha_scale: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
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


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path, quality=95)


def save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)


def spaced_grayscale_values(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [255]
    return [
        int(round(100 + idx * 155 / (count - 1)))
        for idx in range(count)
    ]


def bbox_center(box: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = box
    return [float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)]


def affine_2x3(matrix: np.ndarray) -> list[list[float]]:
    return [
        [float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])],
        [float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])],
    ]


def estimate_hidden_position(proposal: Proposal, frame_idx: int) -> tuple[int, int]:
    visible_indices = [
        idx for idx, box in enumerate(proposal.bboxes)
        if box[2] > box[0] and box[3] > box[1]
    ]
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
        prev_idx = max(idx for idx in visible_indices if idx < frame_idx)
        next_idx = min(idx for idx in visible_indices if idx > frame_idx)
        i0, i1 = prev_idx, next_idx

    x0, y0 = proposal.positions[i0]
    x1, y1 = proposal.positions[i1]
    denom = max(i1 - i0, 1)
    t = (frame_idx - i0) / denom
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    return (int(round(x)), int(round(y)))


def frame_geometry_state(
    overlay: PreparedOverlay,
    proposal: Proposal,
    frame_idx: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    variant_rgb, variant_alpha = get_overlay_variant(overlay, frame_idx)
    del variant_rgb
    forward = temporal_forward_matrix(
        overlay.temporal_mode,
        overlay.temporal_params,
        frame_idx,
        variant_alpha,
    )

    box = proposal.bboxes[frame_idx]
    crop_box = proposal.crop_boxes[frame_idx]
    if box[2] > box[0] and box[3] > box[1] and crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        x, y = proposal.positions[frame_idx]
        bx0, by0, _, _ = crop_box
        placement = np.array(
            [
                [1.0, 0.0, float(x - bx0)],
                [0.0, 1.0, float(y - by0)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return placement @ forward, box

    visible_indices = [
        idx for idx, ref_box in enumerate(proposal.bboxes)
        if ref_box[2] > ref_box[0] and ref_box[3] > ref_box[1]
    ]
    if not visible_indices:
        return np.eye(3, dtype=np.float32), (0, 0, 0, 0)

    ref_idx = min(visible_indices, key=lambda idx: abs(idx - frame_idx))
    ref_box = proposal.bboxes[ref_idx]
    ref_crop = proposal.crop_boxes[ref_idx]
    bw = ref_box[2] - ref_box[0]
    bh = ref_box[3] - ref_box[1]
    x, y = estimate_hidden_position(proposal, frame_idx)
    bx0, by0, _, _ = ref_crop
    placement = np.array(
        [
            [1.0, 0.0, float(x - bx0)],
            [0.0, 1.0, float(y - by0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return placement @ forward, (x, y, x + bw, y + bh)


def overlay_export_record(
    overlay: PreparedOverlay,
    proposal: Proposal,
    mask_paths: list[str],
    alpha_values: list[float],
) -> dict:
    mid_idx = len(alpha_values) // 2
    last_idx = len(alpha_values) - 1
    world_0, box0 = frame_geometry_state(overlay, proposal, 0)
    world_mid, box_mid = frame_geometry_state(overlay, proposal, mid_idx)
    world_last, box_last = frame_geometry_state(overlay, proposal, last_idx)
    affine_0_to_mid = world_mid @ np.linalg.inv(world_0)
    affine_0_to_last = world_last @ np.linalg.inv(world_0)
    affine_2_to_mid = world_mid @ np.linalg.inv(world_last)
    mode = overlay.temporal_mode

    alpha0 = float(alpha_values[0])
    alpha1 = float(alpha_values[mid_idx])
    alpha2 = float(alpha_values[last_idx])
    if mode == "blink":
        if alpha0 > alpha2:
            alpha = (alpha0 + alpha2) / (2.0 * alpha0) if alpha0 > 0.0 else 0.0
            beta = 0.0
        else:
            alpha = 0.0
            beta = (alpha0 + alpha2) / (2.0 * alpha2) if alpha2 > 0.0 else 0.0
    else:
        alpha, beta = 1.0, 0.0

    return {
        "object_id": f"{overlay.overlay_id:03d}",
        "mask_paths": mask_paths,
        "alpha": alpha,
        "beta": beta,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "mode": mode,
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


def sequence_paths() -> list[Path]:
    return sorted(path for path in cfg.SOURCE_ROOT.glob("*/*") if path.is_dir())


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_sequence(
    seq_dir: Path,
    out_dir: Path,
    catalog: dict[str, list[CatalogEntry]],
    font_paths: list[Path],
    rng: random.Random,
) -> dict:
    if cfg.SEQUENCE_LENGTH != 3:
        raise SystemExit("This output format currently supports triplets only (SEQUENCE_LENGTH=3).")

    frame_paths = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    frames = [load_rgb(path) for path in frame_paths]
    if not frames:
        return {"augmented": False, "overlays": []}

    ensure_empty_dir(out_dir)
    overlay_masks_root = out_dir / "overlays_masks"
    aggregate_root = out_dir / "aggregate_masks"
    overlay_masks_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    augmented = rng.random() < cfg.p_augment
    overlays_meta = []
    aggregate_masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
    out_frames = [frame.copy() for frame in frames]

    if augmented:
        prepared: list[PreparedOverlay] = []
        overlay_count = choose_overlay_count(rng)
        for overlay_id in range(overlay_count):
            asset = sample_overlay_asset(catalog, rng)
            overlay = prepare_overlay(
                asset,
                (frames[0].shape[1], frames[0].shape[0]),
                rng,
                overlay_id,
                font_paths,
            )
            overlay.proposals = generate_proposals(frames[0], overlay, rng)
            prepared.append(overlay)

        chosen = select_non_overlapping(prepared)
        grayscale_values = spaced_grayscale_values(len(chosen))
        rng.shuffle(grayscale_values)
        for overlay, proposal in chosen:
            overlay.color = grayscale_values.pop()
            blink_seed = rng.randint(0, 10_000)
            blink_alphas = sample_blink_alphas(rng) if overlay.temporal_mode == "blink" else None
            if blink_alphas is not None:
                overlay.asset.meta["blink_alphas"] = [float(blink_alphas[0]), float(blink_alphas[1])]
            mask_paths: list[str] = []
            alpha_values: list[float] = []

            for frame_idx, base_frame in enumerate(out_frames):
                mode_factor = temporal_alpha_factor(overlay.temporal_mode, frame_idx, blink_seed, blink_alphas)
                alpha_values.append(float(overlay.alpha_scale * mode_factor))
                mask_name = f"{overlay.overlay_id:03d}_M{frame_idx}.png"
                mask_path = overlay_masks_root / mask_name
                if mode_factor == 0.0:
                    save_gray(mask_path, np.zeros(base_frame.shape[:2], dtype=np.uint8))
                    mask_paths.append(mask_name)
                    continue

                variant_rgb, variant_alpha = get_overlay_variant(overlay, frame_idx)
                variant_rgb, variant_alpha = apply_temporal_transform(
                    variant_rgb,
                    variant_alpha,
                    overlay.temporal_mode,
                    overlay.temporal_params,
                    frame_idx,
                )
                bx0, by0, bx1, by1 = proposal.crop_boxes[frame_idx]
                variant_rgb = variant_rgb[by0:by1, bx0:bx1]
                variant_alpha = variant_alpha[by0:by1, bx0:bx1]
                x, y = proposal.positions[frame_idx]
                out_frames[frame_idx], mask, clipped_box = composite(
                    base_frame,
                    variant_rgb,
                    variant_alpha,
                    x,
                    y,
                    alpha_scale=overlay.alpha_scale * mode_factor,
                )

                if mask.size == 0:
                    save_gray(mask_path, np.zeros(base_frame.shape[:2], dtype=np.uint8))
                    mask_paths.append(mask_name)
                    continue

                x0, y0, x1, y1 = clipped_box
                full_mask = np.zeros(base_frame.shape[:2], dtype=np.uint8)
                full_mask[y0:y1, x0:x1] = mask
                patch = aggregate_masks[frame_idx][y0:y1, x0:x1]
                patch[mask > 0] = overlay.color
                save_gray(mask_path, full_mask)
                mask_paths.append(mask_name)

            overlays_meta.append(overlay_export_record(overlay, proposal, mask_paths, alpha_values))

        out_frames = apply_global_seq_aug(out_frames)

    frame_ext = frame_paths[0].suffix.lower() if frame_paths else ".jpg"
    for idx, image in enumerate(out_frames):
        save_rgb(out_dir / f"I{idx}{frame_ext}", image)
        Image.fromarray(aggregate_masks[idx], mode="L").save(aggregate_root / f"M{idx}.png")

    (out_dir / "overlays.json").write_text(json.dumps(overlays_meta, indent=2))
    return {
        "sequence": str(seq_dir.relative_to(cfg.SOURCE_ROOT)),
        "augmented": augmented and bool(overlays_meta),
        "overlay_count": len(overlays_meta),
        "overlays": overlays_meta,
    }


def main() -> None:
    if cfg.SEQUENCE_LENGTH != 3:
        raise SystemExit(f"SEQUENCE_LENGTH must be 3 for this output format, got {cfg.SEQUENCE_LENGTH}")
    rng = random.Random(cfg.SEED)
    catalog = load_catalog()
    font_paths = sample_font_paths(catalog)
    sequences = sequence_paths()
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    augmented_count = 0
    overlay_total = 0
    for seq_dir in sequences:
        rel = seq_dir.relative_to(cfg.SOURCE_ROOT)
        out_dir = cfg.OUTPUT_ROOT / rel
        metadata = build_sequence(seq_dir, out_dir, catalog, font_paths, rng)
        if metadata["augmented"]:
            augmented_count += 1
            overlay_total += len(metadata["overlays"])

    print(
        f"Processed {len(sequences)} sequences into {cfg.OUTPUT_ROOT} "
        f"with {augmented_count} augmented sequences and {overlay_total} placed overlays."
    )


if __name__ == "__main__":
    main()
