#!/usr/bin/env python3
"""Build a Vimeo triplet augmentation dataset with detailed task metadata."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from Inconsistencies import build_dataset as inc  # noqa: E402
from vimeo_triplet_aug import config as cfg  # noqa: E402


inc.cfg = cfg

IMAGE_EXTS = inc.IMAGE_EXTS


@dataclass
class VimeoSequenceItem:
    split_name: str
    rel: str
    path: Path


@dataclass
class VimeoPreparedOverlay:
    overlay_id: int
    asset: inc.OverlayAsset
    frame_images: list[np.ndarray]
    frame_alphas: list[np.ndarray]
    proposal_frame_images: list[np.ndarray]
    proposal_frame_alphas: list[np.ndarray]
    copied_mid_image: np.ndarray
    copied_mid_alpha: np.ndarray
    geometry_mode: str
    geometry_submode: str
    geometry_params: dict
    motion_step: tuple[float, float]
    alpha_scales: list[float]
    copied_mid_alpha_scale: float
    temporal_mode: str
    temporal_detail: dict
    discontinuity_label: int
    color: int
    proposals: list[inc.Proposal]


def weighted_choice(items, weights, rng: random.Random):
    return inc.weighted_choice(items, weights, rng)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)


def load_catalog(split_file: Path) -> dict[str, list[inc.CatalogEntry]]:
    by_class: dict[str, list[inc.CatalogEntry]] = {}
    if not split_file.exists():
        return by_class
    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
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
            inc.CatalogEntry(
                cls_name=cls_name,
                subfolder=rel,
                directory=directory,
                files=sorted(files),
                weight=weight,
            )
        )
    return by_class


def goons_class_weight(cls_name: str) -> float:
    return float(cfg.GOONS_CLASS_WEIGHTS.get(cls_name, 1.0))


def choose_overlay_count(rng: random.Random) -> int:
    lo, hi = weighted_choice(cfg.count_ranges, cfg.count_ranges_prob, rng)
    return rng.randint(lo, hi)


def sample_class(catalog: dict[str, list[inc.CatalogEntry]], rng: random.Random) -> str:
    classes = [cls_name for cls_name, weight in cfg.GOONS_CLASS_WEIGHTS.items() if weight > 0.0 and catalog.get(cls_name)]
    weights = [goons_class_weight(cls_name) for cls_name in classes]
    return weighted_choice(classes, weights, rng)


def sample_catalog_entry(catalog: dict[str, list[inc.CatalogEntry]], cls_name: str, rng: random.Random) -> inc.CatalogEntry:
    entries = catalog[cls_name]
    return weighted_choice(entries, [entry.weight for entry in entries], rng)


def sample_overlay_asset_for_class(catalog: dict[str, list[inc.CatalogEntry]], cls_name: str, rng: random.Random) -> inc.OverlayAsset:
    entry = sample_catalog_entry(catalog, cls_name, rng)
    source_path = rng.choice(entry.files)
    if cls_name == "Font":
        image, alpha, font_meta = inc.render_font_asset(source_path, rng)
        return inc.OverlayAsset(
            cls_name=cls_name,
            source_rel=str(source_path.relative_to(cfg.GOONS_ROOT)),
            image=image,
            alpha=alpha,
            meta={"subfolder": entry.subfolder, **font_meta},
        )
    image, alpha = inc.load_rgba_asset(source_path)
    return inc.OverlayAsset(
        cls_name=cls_name,
        source_rel=str(source_path.relative_to(cfg.GOONS_ROOT)),
        image=image,
        alpha=alpha,
        meta={"subfolder": entry.subfolder},
    )


def sample_overlay_asset(catalog: dict[str, list[inc.CatalogEntry]], rng: random.Random) -> inc.OverlayAsset:
    cls_name = sample_class(catalog, rng)
    return sample_overlay_asset_for_class(catalog, cls_name, rng)


def list_split_items(split_name: str, list_path: Path, limit: int | None = None) -> list[VimeoSequenceItem]:
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if limit is not None:
        lines = lines[:limit]
    return [
        VimeoSequenceItem(
            split_name=split_name,
            rel=rel,
            path=cfg.VIMEO_SEQUENCE_ROOT / rel,
        )
        for rel in lines
    ]


def load_vimeo_frames(seq_dir: Path) -> tuple[list[np.ndarray], str]:
    frame_paths = [seq_dir / "im1.png", seq_dir / "im2.png", seq_dir / "im3.png"]
    for path in frame_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Vimeo frame: {path}")
    return [inc.load_rgb(path) for path in frame_paths], ".png"


def sample_sequence_profile(rng: random.Random) -> dict:
    geometry_mode = weighted_choice(
        list(cfg.GEOMETRY_MODE_PROBS.keys()),
        list(cfg.GEOMETRY_MODE_PROBS.values()),
        rng,
    )
    geometry_submode = "static"
    if geometry_mode == "motion":
        geometry_submode = weighted_choice(
            list(cfg.MOTION_SUBMODE_PROBS.keys()),
            list(cfg.MOTION_SUBMODE_PROBS.values()),
            rng,
        )
    temporal_mode = weighted_choice(
        list(cfg.TEMPORAL_MODE_PROBS.keys()),
        list(cfg.TEMPORAL_MODE_PROBS.values()),
        rng,
    )
    detail: dict = {}
    if temporal_mode == "appear_disappear":
        detail["variant"] = "disappear" if rng.random() < 0.5 else "appear"
    elif temporal_mode == "change_appearance":
        variant = weighted_choice(
            list(cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS.keys()),
            list(cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS.values()),
            rng,
        )
        if variant == "composite":
            primitive_modes = ["color", "visibility", "textual"]
            primitive_weights = [
                cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS["color"],
                cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS["visibility"],
                cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS["textual"],
            ]
            first = weighted_choice(primitive_modes, primitive_weights, rng)
            remaining_modes = [mode for mode in primitive_modes if mode != first]
            remaining_weights = [
                cfg.CHANGE_APPEARANCE_COMPONENT_WEIGHTS[mode]
                for mode in remaining_modes
            ]
            second = weighted_choice(remaining_modes, remaining_weights, rng)
            detail["variant"] = "composite"
            detail["components"] = [first, second]
        else:
            detail["variant"] = variant
            detail["components"] = [variant]
    return {
        "geometry_mode": geometry_mode,
        "geometry_submode": geometry_submode,
        "transparent": rng.random() < cfg.TRANSPARENT_PROB,
        "temporal_mode": temporal_mode,
        "temporal_detail": detail,
    }


def blend_images(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    return np.clip((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32), 0, 255).astype(np.uint8)


def sample_color_change_params(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = cfg.appearance_change_color_scale_range
    factors = np.array(
        [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)],
        dtype=np.float32,
    )
    perm = np.array(rng.sample([0, 1, 2], 3), dtype=np.int64)
    return factors, perm


def apply_color_change_params(image: np.ndarray, factors: np.ndarray, perm: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * factors[None, None, :], 0, 255).astype(np.uint8)
    return out[..., perm]


def sample_color_scale_params(rng: random.Random, lo: float, hi: float) -> np.ndarray:
    return np.array(
        [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)],
        dtype=np.float32,
    )


def apply_color_scale_params(image: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factors[None, None, :], 0, 255).astype(np.uint8)


def apply_color_change_variant(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    factors, perm = sample_color_change_params(rng)
    alt_image = apply_color_change_params(image.copy(), factors, perm)
    return alt_image, {
        "variant": "color",
        "color_multipliers": factors.tolist(),
        "color_permutation": perm.tolist(),
    }


def apply_visibility_change_variant(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    start_alpha = rng.uniform(0.6, 1.0)
    end_alpha = rng.uniform(0.6, 1.0)
    mid_alpha = 0.5 * (start_alpha + end_alpha)
    detail = {
        "variant": "visibility",
        "visibility_variant": "transparency",
        "alpha0": start_alpha,
        "alpha05": mid_alpha,
        "alpha1": end_alpha,
        "alpha_scales": [start_alpha, mid_alpha, end_alpha],
    }
    return image.copy(), detail


def build_appear_disappear_triplet(image: np.ndarray, alpha: np.ndarray, rng: random.Random, variant: str) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict]:
    blank_img = np.zeros_like(image)
    blank_alpha = np.zeros_like(alpha)
    if variant == "disappear":
        return [image.copy(), image.copy(), blank_img], [alpha.copy(), alpha.copy(), blank_alpha], image.copy(), alpha.copy(), {"variant": "disappear"}
    return [blank_img, blank_img, image.copy()], [blank_alpha, blank_alpha, alpha.copy()], blank_img, blank_alpha, {"variant": "appear"}


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


def add_text_within_mask(
    base_image: np.ndarray,
    alpha: np.ndarray,
    font_path: Path,
    text: str,
    rng: random.Random,
    fill: tuple[int, int, int] = (255, 255, 255),
    stroke_width: int | None = None,
) -> np.ndarray:
    bbox = inc.alpha_bbox(alpha)
    if bbox is None:
        return base_image.copy()

    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
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


def build_font_texture_change(
    asset: inc.OverlayAsset,
    image: np.ndarray,
    alpha: np.ndarray,
    aug_replay: dict,
    new_size: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    font_path = cfg.GOONS_ROOT / asset.source_rel
    base_text = asset.meta.get("text", "")
    if not base_text:
        base_text = random_text(rng)
    alt_text = random_text_same_length(base_text, rng)
    fill = tuple(asset.meta.get("fill", cfg.font_fill_palette[0]))
    stroke_width = asset.meta.get("stroke_width")
    font_size = asset.meta.get("font_size")
    raw_alt_image, raw_alt_alpha, _ = inc.render_font_asset(
        font_path,
        rng,
        canvas_size=(asset.image.shape[1], asset.image.shape[0]),
        text=alt_text,
        font_size=font_size,
        fill=fill,
        stroke_width=stroke_width,
    )
    alt_image, alt_alpha, _ = inc.apply_overlay_aug(raw_alt_image, raw_alt_alpha, replay=aug_replay)
    alt_image, alt_alpha = inc.resize_overlay(alt_image, alt_alpha, new_size)
    return image.copy(), alpha.copy(), alt_image, alt_alpha, {
        "variant": "textual",
        "textual_variant": "font_text_to_text",
        "text_transition_mode": "text_to_text",
        "text_before": base_text,
        "text_after": alt_text,
    }


def build_nonfont_texture_change(
    image: np.ndarray,
    alpha: np.ndarray,
    font_paths: list[Path],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    font_path = rng.choice(font_paths)
    transition_mode = sample_text_transition_mode(rng)
    base_text = random_text(rng)
    alt_text = random_text_same_length(base_text, rng)
    fill = (255, 255, 255)
    stroke_width = rng.randint(*cfg.font_stroke_width)

    base_image = image.copy()
    alt_image = image.copy()
    if transition_mode == "none_to_text":
        base_text = ""
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)
    elif transition_mode == "text_to_none":
        alt_text = ""
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
    else:
        base_image = add_text_within_mask(base_image, alpha, font_path, base_text, rng, fill=fill, stroke_width=stroke_width)
        alt_image = add_text_within_mask(alt_image, alpha, font_path, alt_text, rng, fill=fill, stroke_width=stroke_width)

    detail = {
        "variant": "textual",
        "textual_variant": "nonfont_text_change",
        "text_transition_mode": transition_mode,
        "text_before": base_text,
        "text_after": alt_text,
        "text_font": str(font_path.relative_to(cfg.GOONS_ROOT)),
        "text_fill": list(fill),
        "text_stroke_width": stroke_width,
    }
    return base_image, alpha.copy(), alt_image, alpha.copy(), detail


def build_texture_change_variant(asset: inc.OverlayAsset, image: np.ndarray, alpha: np.ndarray, aug_replay: dict, new_size: tuple[int, int], font_paths: list[Path], rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if asset.cls_name == "Font":
        return build_font_texture_change(asset, image, alpha, aug_replay, new_size, rng)
    return build_nonfont_texture_change(image, alpha, font_paths, rng)


def build_change_appearance_triplet(
    asset: inc.OverlayAsset,
    image: np.ndarray,
    alpha: np.ndarray,
    aug_replay: dict,
    new_size: tuple[int, int],
    font_paths: list[Path],
    rng: random.Random,
    appearance_detail: dict,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict]:
    components = list(appearance_detail.get("components", []))
    base_image = image.copy()
    base_alpha = alpha.copy()
    mid_image = image.copy()
    mid_alpha = alpha.copy()
    end_image = image.copy()
    end_alpha = alpha.copy()
    copied_mid = image.copy()
    copied_mid_alpha = alpha.copy()
    detail = {"variant": appearance_detail.get("variant"), "components": components, "component_details": []}
    alpha_scales = None

    if "textual" in components:
        base_image, base_alpha, end_image, end_alpha, texture_detail = build_texture_change_variant(
            asset, image, alpha, aug_replay, new_size, font_paths, rng
        )
        mid_image = base_image.copy()
        mid_alpha = base_alpha.copy()
        copied_mid = base_image.copy()
        copied_mid_alpha = base_alpha.copy()
        detail["component_details"].append(texture_detail)
    if "color" in components:
        factors, perm = sample_color_change_params(rng)
        end_image = apply_color_change_params(end_image, factors, perm)
        mid_image = blend_images(mid_image, apply_color_change_params(mid_image, factors, perm))
        color_detail = {
            "variant": "color",
            "color_multipliers": factors.tolist(),
            "color_permutation": perm.tolist(),
        }
        detail["component_details"].append(color_detail)
    if "visibility" in components:
        start_alpha = rng.uniform(0.6, 1.0)
        end_alpha_scale = rng.uniform(0.6, 1.0)
        mid_alpha_scale = 0.5 * (start_alpha + end_alpha_scale)
        alpha_scales = [start_alpha, mid_alpha_scale, end_alpha_scale]
        visibility_detail = {
            "variant": "visibility",
            "visibility_variant": "transparency",
            "alpha0": start_alpha,
            "alpha05": mid_alpha_scale,
            "alpha1": end_alpha_scale,
            "alpha_scales": alpha_scales,
        }
        detail["component_details"].append(visibility_detail)
    detail["alpha_scales"] = alpha_scales
    return [base_image, mid_image, end_image], [base_alpha, mid_alpha, end_alpha], copied_mid, copied_mid_alpha, detail


def build_none_triplet(image: np.ndarray, alpha: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict]:
    return [image.copy(), image.copy(), image.copy()], [alpha.copy(), alpha.copy(), alpha.copy()], image.copy(), alpha.copy(), {}


def prepare_overlay(
    asset: inc.OverlayAsset,
    frame_size: tuple[int, int],
    catalog: dict[str, list[inc.CatalogEntry]],
    font_paths: list[Path],
    profile: dict,
    rng: random.Random,
    overlay_id: int,
) -> VimeoPreparedOverlay:
    del catalog
    frame_w, frame_h = frame_size
    image, alpha, aug_replay = inc.apply_overlay_aug(asset.image, asset.alpha)
    frame_area = float(frame_w * frame_h)
    overlay_area = float(image.shape[0] * image.shape[1])
    target_area_ratio = rng.uniform(cfg.min_area_ratio, cfg.max_area_ratio)
    scale = (target_area_ratio * frame_area / max(overlay_area, 1.0)) ** 0.5
    new_w = max(8, int(round(image.shape[1] * scale)))
    new_h = max(8, int(round(image.shape[0] * scale)))
    image, alpha = inc.resize_overlay(image, alpha, (new_w, new_h))

    temporal_mode = profile["temporal_mode"]
    if temporal_mode == "none":
        frame_images, frame_alphas, copied_mid_image, copied_mid_alpha, temporal_detail = build_none_triplet(image, alpha)
    elif temporal_mode == "change_appearance":
        frame_images, frame_alphas, copied_mid_image, copied_mid_alpha, temporal_detail = build_change_appearance_triplet(
            asset, image, alpha, aug_replay, (new_w, new_h), font_paths, rng, profile["temporal_detail"]
        )
    elif temporal_mode == "appear_disappear":
        frame_images, frame_alphas, copied_mid_image, copied_mid_alpha, temporal_detail = build_appear_disappear_triplet(
            image, alpha, rng, profile["temporal_detail"]["variant"]
        )
    else:
        raise RuntimeError(f"Unsupported temporal mode: {temporal_mode}")

    geometry_mode = "static"
    geometry_params: dict = {}
    motion_step = (0.0, 0.0)
    if profile["geometry_mode"] == "motion":
        submode = profile["geometry_submode"]
        if submode in {"translational", "both"}:
            motion_step = inc.sample_motion_step(rng)
        if submode in {"affine", "both"}:
            geometry_mode = "affine"
            geometry_params = inc.sample_affine_params(rng)
    geometry_submode = profile["geometry_submode"]

    alpha_scales = [1.0, 1.0, 1.0]
    copied_mid_alpha_scale = 1.0
    if profile["transparent"]:
        alpha_scales = [cfg.transparent_alpha] * 3
        copied_mid_alpha_scale = cfg.transparent_alpha
    if temporal_mode == "change_appearance" and temporal_detail.get("alpha_scales") is not None:
        alpha_scales = temporal_detail["alpha_scales"]
        copied_mid_alpha_scale = alpha_scales[0]

    proposal_frame_images = [frame_images[0].copy(), copied_mid_image.copy(), frame_images[2].copy()]
    proposal_frame_alphas = [frame_alphas[0].copy(), copied_mid_alpha.copy(), frame_alphas[2].copy()]

    if geometry_mode == "affine":
        canvas_w, canvas_h = inc.affine_canvas_size(frame_images[0].shape[1], frame_images[0].shape[0], geometry_params)
    else:
        canvas_w, canvas_h = frame_images[0].shape[1], frame_images[0].shape[0]

    centered_images = []
    centered_alphas = []
    for img, alp in zip(frame_images, frame_alphas):
        ci, ca = inc.center_on_canvas(img, alp, (canvas_w, canvas_h))
        centered_images.append(ci)
        centered_alphas.append(ca)
    frame_images = centered_images
    frame_alphas = centered_alphas

    centered_prop_images = []
    centered_prop_alphas = []
    for img, alp in zip(proposal_frame_images, proposal_frame_alphas):
        ci, ca = inc.center_on_canvas(img, alp, (canvas_w, canvas_h))
        centered_prop_images.append(ci)
        centered_prop_alphas.append(ca)
    proposal_frame_images = centered_prop_images
    proposal_frame_alphas = centered_prop_alphas

    copied_mid_image, copied_mid_alpha = inc.center_on_canvas(copied_mid_image, copied_mid_alpha, (canvas_w, canvas_h))

    return VimeoPreparedOverlay(
        overlay_id=overlay_id,
        asset=asset,
        frame_images=frame_images,
        frame_alphas=frame_alphas,
        proposal_frame_images=proposal_frame_images,
        proposal_frame_alphas=proposal_frame_alphas,
        copied_mid_image=copied_mid_image,
        copied_mid_alpha=copied_mid_alpha,
        geometry_mode=geometry_mode,
        geometry_submode=geometry_submode,
        geometry_params=geometry_params,
        motion_step=motion_step,
        alpha_scales=alpha_scales,
        copied_mid_alpha_scale=copied_mid_alpha_scale,
        temporal_mode=temporal_mode,
        temporal_detail=temporal_detail,
        discontinuity_label=1 if temporal_mode == "appear_disappear" or "textual" in temporal_detail.get("components", []) else 0,
        color=0,
        proposals=[],
    )


def frame_variants_for_proposals(overlay: VimeoPreparedOverlay) -> list[dict | None]:
    variants: list[dict | None] = []
    for frame_idx in range(3):
        rgb, alpha = inc.apply_temporal_transform(
            overlay.proposal_frame_images[frame_idx],
            overlay.proposal_frame_alphas[frame_idx],
            overlay.geometry_mode,
            overlay.geometry_params,
            frame_idx,
        )
        bbox = inc.alpha_bbox(alpha)
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


def generate_proposals(frame: np.ndarray, overlay: VimeoPreparedOverlay, rng: random.Random) -> list[inc.Proposal]:
    proposals: list[inc.Proposal] = []
    frame_h, frame_w = frame.shape[:2]
    frame_variants = frame_variants_for_proposals(overlay)
    visible_indices = [idx for idx, item in enumerate(frame_variants) if item is not None]
    step_x = overlay.motion_step[0] * frame_w
    step_y = overlay.motion_step[1] * frame_w
    first_visible_idx = visible_indices[0] if visible_indices else None
    if first_visible_idx is None:
        return proposals

    def clipped_contrast_score(image: np.ndarray, alpha: np.ndarray, x: int, y: int) -> float | None:
        h, w = alpha.shape
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame_w, x + w)
        y1 = min(frame_h, y + h)
        if x0 >= x1 or y0 >= y1:
            return None
        src_x0 = x0 - x
        src_y0 = y0 - y
        src_x1 = src_x0 + (x1 - x0)
        src_y1 = src_y0 + (y1 - y0)
        image_clip = image[src_y0:src_y1, src_x0:src_x1]
        alpha_clip = alpha[src_y0:src_y1, src_x0:src_x1]
        return inc.contrast_score(frame, image_clip, alpha_clip, x0, y0)

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
        for frame_idx in range(3):
            variant = frame_variants[frame_idx]
            blocker_variant = variant
            if blocker_variant is None and overlay.temporal_mode == "appear_disappear" and visible_indices:
                ref_idx = min(visible_indices, key=lambda idx: abs(idx - frame_idx))
                blocker_variant = frame_variants[ref_idx]
            if variant is None:
                if blocker_variant is None:
                    positions.append((0, 0))
                    bboxes.append((0, 0, 0, 0))
                    crop_boxes.append((0, 0, 0, 0))
                    masks.append(np.zeros((0, 0), dtype=bool))
                    mask_areas.append(0)
                    continue
            if blocker_variant is None:
                positions.append((0, 0))
                bboxes.append((0, 0, 0, 0))
                crop_boxes.append((0, 0, 0, 0))
                masks.append(np.zeros((0, 0), dtype=bool))
                mask_areas.append(0)
                continue
            bx0, by0, bx1, by1 = blocker_variant["bbox"]
            bw = bx1 - bx0
            bh = by1 - by0
            if bw <= 0 or bh <= 0:
                valid = False
                break
            if not anchor_initialized:
                max_left = frame_w - 1
                max_top = frame_h - 1
                min_left = 1 - bw
                min_top = 1 - bh
                if min_left > max_left or min_top > max_top:
                    valid = False
                    break
                anchor_x0 = rng.randint(min_left, max_left) - int(round(step_x * frame_idx))
                anchor_y0 = rng.randint(min_top, max_top) - int(round(step_y * frame_idx))
                anchor_initialized = True
            x = int(round(anchor_x0 + step_x * frame_idx))
            y = int(round(anchor_y0 + step_y * frame_idx))
            if x + bw <= 0 or y + bh <= 0 or x >= frame_w or y >= frame_h:
                valid = False
                break
            positions.append((x, y))
            bboxes.append((x, y, x + bw, y + bh))
            crop_boxes.append((bx0, by0, bx1, by1))
            masks.append(blocker_variant["mask"])
            mask_areas.append(blocker_variant["mask_area"])
        if not valid:
            continue
        first_variant = frame_variants[first_visible_idx]
        bx0, by0, bx1, by1 = crop_boxes[first_visible_idx]
        score = clipped_contrast_score(
            first_variant["rgb"][by0:by1, bx0:bx1],
            first_variant["alpha"][by0:by1, bx0:bx1],
            bboxes[first_visible_idx][0],
            bboxes[first_visible_idx][1],
        )
        if score is None:
            continue
        proposals.append(
            inc.Proposal(
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


def select_non_overlapping(overlays: list[VimeoPreparedOverlay]) -> list[tuple[VimeoPreparedOverlay, inc.Proposal]]:
    ordered = sorted(
        overlays,
        key=lambda item: (
            -(item.proposals[0].score if item.proposals else -1.0),
            len(item.proposals),
            -(item.frame_alphas[0].shape[0] * item.frame_alphas[0].shape[1]),
        ),
    )
    chosen: list[tuple[VimeoPreparedOverlay, inc.Proposal]] = []
    for overlay in ordered:
        for proposal in overlay.proposals:
            if any(inc.proposals_conflict(proposal, existing) for _, existing in chosen):
                continue
            chosen.append((overlay, proposal))
            break
    return chosen


def affine_2x3(matrix: np.ndarray) -> list[list[float]]:
    return inc.affine_2x3(matrix)


def bbox_center(box: tuple[int, int, int, int]) -> list[float]:
    return inc.bbox_center(box)


def estimate_hidden_position(proposal: inc.Proposal, frame_idx: int) -> tuple[int, int]:
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


def frame_geometry_state(overlay: VimeoPreparedOverlay, proposal: inc.Proposal, frame_idx: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    forward = inc.temporal_forward_matrix(
        overlay.geometry_mode,
        overlay.geometry_params,
        frame_idx,
        overlay.proposal_frame_alphas[frame_idx],
    )
    box = proposal.bboxes[frame_idx]
    crop_box = proposal.crop_boxes[frame_idx]
    if box[2] > box[0] and box[3] > box[1]:
        x, y = proposal.positions[frame_idx]
        bx0, by0, _, _ = crop_box
        placement = np.array([[1.0, 0.0, float(x - bx0)], [0.0, 1.0, float(y - by0)], [0.0, 0.0, 1.0]], dtype=np.float32)
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
        [[1.0, 0.0, float(x - bx0)], [0.0, 1.0, float(y - by0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return placement @ forward, (x, y, x + bw, y + bh)


def overlay_metadata_record(overlay: VimeoPreparedOverlay, proposal: inc.Proposal, mask_paths: dict[str, str]) -> dict:
    world_0, box0 = frame_geometry_state(overlay, proposal, 0)
    world_05, box05 = frame_geometry_state(overlay, proposal, 1)
    world_1, box1 = frame_geometry_state(overlay, proposal, 2)
    affine_0_to_05 = world_05 @ np.linalg.inv(world_0)
    affine_0_to_1 = world_1 @ np.linalg.inv(world_0)
    affine_1_to_05 = world_05 @ np.linalg.inv(world_1)

    alpha0 = float(overlay.alpha_scales[0])
    alpha05 = float(overlay.alpha_scales[1])
    alpha1 = float(overlay.alpha_scales[2])
    if overlay.temporal_mode == "change_appearance" and overlay.temporal_detail.get("alpha_scales") is not None:
        if alpha0 > alpha1:
            alpha = (alpha0 + alpha1) / (2.0 * alpha0) if alpha0 > 0.0 else 0.0
            beta = 0.0
        else:
            alpha = 0.0
            beta = (alpha0 + alpha1) / (2.0 * alpha1) if alpha1 > 0.0 else 0.0
    else:
        alpha = 1.0
        beta = 0.0

    return {
        "object_id": f"{overlay.overlay_id:03d}",
        "mode": overlay.temporal_mode,
        "A": affine_2x3(affine_0_to_05),
        "B": affine_2x3(affine_1_to_05),
        "alpha": alpha,
        "beta": beta,
        "alpha0": alpha0,
        "alpha05": alpha05,
        "alpha1": alpha1,
        "goons": {
            "class_name": overlay.asset.cls_name,
            "class_weight": goons_class_weight(overlay.asset.cls_name),
            "source_rel": overlay.asset.source_rel,
            "subfolder": overlay.asset.meta.get("subfolder"),
        },
        "geometry": {
            "mode": overlay.geometry_mode,
            "submode": overlay.geometry_submode,
            "params": overlay.geometry_params,
            "motion_step": [float(overlay.motion_step[0]), float(overlay.motion_step[1])],
            "affine_0_to_05": affine_2x3(affine_0_to_05),
            "affine_1_to_05": affine_2x3(affine_1_to_05),
            "affine_0_to_1": affine_2x3(affine_0_to_1),
            "bbox0": list(map(int, box0)),
            "bbox05": list(map(int, box05)),
            "bbox1": list(map(int, box1)),
            "center0": bbox_center(box0),
            "center05": bbox_center(box05),
            "center1": bbox_center(box1),
        },
        "transparent": {
            "enabled": any(abs(alpha - 1.0) > 1e-6 for alpha in overlay.alpha_scales) or abs(overlay.copied_mid_alpha_scale - 1.0) > 1e-6,
            "alpha_scales": {
                "I0": float(overlay.alpha_scales[0]),
                "I_0.5": float(overlay.alpha_scales[1]),
                "I1": float(overlay.alpha_scales[2]),
                "I_0.5_copied": float(overlay.copied_mid_alpha_scale),
            },
        },
        "temporal": {
            "mode": overlay.temporal_mode,
            "detail": overlay.temporal_detail,
            "discontinuity_label": overlay.discontinuity_label,
        },
        "mask_paths": mask_paths,
    }


def base_task_metadata(overlays_meta: list[dict]) -> dict:
    return {
        "segmentation": {
            "description": "input is augmented I, output is tracked masks, loss against augmented masks",
            "input_frames": ["I0.png", "I_0.5.png", "I1.png"],
            "aggregate_masks": {
                "I0": "aggregate_masks/M0.png",
                "I_0.5": "aggregate_masks/M05.png",
                "I1": "aggregate_masks/M1.png",
                "I_0.5_copied": "aggregate_masks/M05_copied.png",
            },
            "tracked_masks": [
                {
                    "object_id": record["object_id"],
                    "masks": record["mask_paths"],
                }
                for record in overlays_meta
            ],
        },
        "classification": {
            "description": "input is two positioned endpoint masks, output is discontinuity 0/1",
            "samples": [
                {
                    "object_id": record["object_id"],
                    "mask_pair": [record["mask_paths"]["I0"], record["mask_paths"]["I1"]],
                    "label": int(record["temporal"]["discontinuity_label"]),
                    "temporal_mode": record["temporal"]["mode"],
                }
                for record in overlays_meta
            ],
        },
        "refinement": {
            "description": "input is I_0.5, target is I_0.5_copied",
            "input_frame": "I_0.5.png",
            "target_frame": "I_0.5_copied.png",
        },
    }


def build_augmented_sequence(
    item: VimeoSequenceItem,
    out_dir: Path,
    catalog: dict[str, list[inc.CatalogEntry]],
    font_paths: list[Path],
    rng: random.Random,
    profile: dict,
) -> dict:
    del profile
    frames, _ = load_vimeo_frames(item.path)
    ensure_empty_dir(out_dir)
    overlay_masks_root = out_dir / "overlays_masks"
    aggregate_root = out_dir / "aggregate_masks"
    overlay_masks_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)
    overlay_count = choose_overlay_count(rng)

    prepared: list[VimeoPreparedOverlay] = []
    for overlay_id in range(overlay_count):
        asset = sample_overlay_asset(catalog, rng)
        overlay_profile = sample_sequence_profile(rng)
        overlay = prepare_overlay(
            asset,
            (frames[0].shape[1], frames[0].shape[0]),
            catalog,
            font_paths,
            overlay_profile,
            rng,
            overlay_id,
        )
        overlay.proposals = generate_proposals(frames[0], overlay, rng)
        if not overlay.proposals:
            continue
        prepared.append(overlay)
    chosen = select_non_overlapping(prepared)

    out_frames = {
        "I0": frames[0].copy(),
        "I_0.5": frames[1].copy(),
        "I1": frames[2].copy(),
        "I_0.5_copied": frames[1].copy(),
    }
    aggregate_masks = {name: np.zeros(frames[0].shape[:2], dtype=np.uint8) for name in out_frames}
    overlays_meta = []
    grayscale_values = inc.spaced_grayscale_values(len(chosen))
    rng.shuffle(grayscale_values)
    for overlay, proposal in chosen:
        overlay.color = grayscale_values.pop()
        mask_paths: dict[str, str] = {}
        frame_name_map = [("I0", 0), ("I_0.5", 1), ("I1", 2)]
        for logical_name, frame_idx in frame_name_map:
            variant_rgb, variant_alpha = inc.apply_temporal_transform(
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
            out_frames[logical_name], mask, clipped_box = inc.composite(
                out_frames[logical_name],
                variant_rgb,
                variant_alpha,
                x,
                y,
                overlay.alpha_scales[frame_idx],
            )
            mask_name = f"{overlay.overlay_id:03d}_{logical_name}.png"
            full_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            if mask.size > 0:
                x0, y0, x1, y1 = clipped_box
                full_mask[y0:y1, x0:x1] = mask
                patch = aggregate_masks[logical_name][y0:y1, x0:x1]
                patch[mask > 0] = overlay.color
            save_gray(overlay_masks_root / mask_name, full_mask)
            mask_paths[logical_name] = f"overlays_masks/{mask_name}"

        # Copied middle uses frame-0 geometry and frame-0 copied-state content.
        copied_rgb, copied_alpha = inc.apply_temporal_transform(
            overlay.copied_mid_image,
            overlay.copied_mid_alpha,
            overlay.geometry_mode,
            overlay.geometry_params,
            0,
        )
        bx0, by0, bx1, by1 = proposal.crop_boxes[0]
        copied_rgb = copied_rgb[by0:by1, bx0:bx1]
        copied_alpha = copied_alpha[by0:by1, bx0:bx1]
        x, y = proposal.positions[0]
        out_frames["I_0.5_copied"], mask, clipped_box = inc.composite(
            out_frames["I_0.5_copied"],
            copied_rgb,
            copied_alpha,
            x,
            y,
            overlay.copied_mid_alpha_scale,
        )
        mask_name = f"{overlay.overlay_id:03d}_I_0.5_copied.png"
        full_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        if mask.size > 0:
            x0, y0, x1, y1 = clipped_box
            full_mask[y0:y1, x0:x1] = mask
            patch = aggregate_masks["I_0.5_copied"][y0:y1, x0:x1]
            patch[mask > 0] = overlay.color
        save_gray(overlay_masks_root / mask_name, full_mask)
        mask_paths["I_0.5_copied"] = f"overlays_masks/{mask_name}"
        overlays_meta.append(overlay_metadata_record(overlay, proposal, mask_paths))

    aug_frames = inc.apply_global_seq_aug(
        [out_frames["I0"], out_frames["I_0.5"], out_frames["I1"], out_frames["I_0.5_copied"]]
    )
    out_frames["I0"], out_frames["I_0.5"], out_frames["I1"], out_frames["I_0.5_copied"] = aug_frames

    save_rgb(out_dir / "I0.png", out_frames["I0"])
    save_rgb(out_dir / "I1.png", out_frames["I1"])
    save_rgb(out_dir / "I_0.5.png", out_frames["I_0.5"])
    save_rgb(out_dir / "I_0.5_copied.png", out_frames["I_0.5_copied"])
    for logical_name, mask_name in [("I0", "M0.png"), ("I_0.5", "M05.png"), ("I1", "M1.png"), ("I_0.5_copied", "M05_copied.png")]:
        save_gray(aggregate_root / mask_name, aggregate_masks[logical_name])

    metadata = {
        "split": item.split_name,
        "source_rel": item.rel,
        "scenechange": None,
        "composite_profile": {
            "scope": "per_overlay",
            "overlay_count_target": overlay_count,
            "overlay_count_placed": len(chosen),
        },
        "tasks": base_task_metadata(overlays_meta),
        "overlays": overlays_meta,
        "objects": overlays_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def make_derangement(size: int, rng: random.Random) -> list[int]:
    if size < 2:
        return list(range(size))
    indices = list(range(size))
    while True:
        perm = indices[:]
        rng.shuffle(perm)
        if all(i != perm[i] for i in indices):
            return perm


def build_scenechange_sequence(item: VimeoSequenceItem, perm_item: VimeoSequenceItem, out_dir: Path, rng: random.Random) -> dict:
    frames_a, _ = load_vimeo_frames(item.path)
    frames_b, _ = load_vimeo_frames(perm_item.path)
    if rng.random() < cfg.SCENECHANGE_PERMUTE_PROB:
        frames = {
            "I0": frames_a[0],
            "I_0.5": frames_a[1],
            "I1": frames_b[2],
            "I_0.5_copied": frames_a[1],
        }
        scenechange_mode = "permutation"
        dark_alpha = None
    else:
        dark_alpha = rng.uniform(*cfg.SCENECHANGE_DARK_RANGE)
        if rng.random() < 0.5:
            frames = {
                "I0": frames_a[0],
                "I_0.5": frames_a[1],
                "I1": inc.apply_dark_overlay(frames_a[2], dark_alpha),
                "I_0.5_copied": frames_a[1],
            }
            scenechange_mode = "darken_third"
        else:
            frames = {
                "I0": inc.apply_dark_overlay(frames_a[0], dark_alpha),
                "I_0.5": inc.apply_dark_overlay(frames_a[1], dark_alpha),
                "I1": frames_a[2],
                "I_0.5_copied": inc.apply_dark_overlay(frames_a[1], dark_alpha),
            }
            scenechange_mode = "darken_first_second"

    ensure_empty_dir(out_dir)
    aggregate_root = out_dir / "aggregate_masks"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays_masks").mkdir(parents=True, exist_ok=True)
    for name, image in frames.items():
        save_rgb(out_dir / f"{name}.png", image)
    for mask_name in ["M0.png", "M05.png", "M1.png", "M05_copied.png"]:
        save_gray(aggregate_root / mask_name, np.zeros(frames["I0"].shape[:2], dtype=np.uint8))

    metadata = {
        "split": item.split_name,
        "source_rel": item.rel,
        "scenechange": {
            "mode": scenechange_mode,
            "dark_alpha": dark_alpha,
            "permuted_source_rel": perm_item.rel,
            "preserved_go": False,
        },
        "composite_profile": None,
        "tasks": base_task_metadata([]),
        "overlays": [],
        "objects": [],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def build_scenechange_preserved_go_sequence(
    item: VimeoSequenceItem,
    perm_item: VimeoSequenceItem,
    out_dir: Path,
    catalog: dict[str, list[inc.CatalogEntry]],
    font_paths: list[Path],
    rng: random.Random,
) -> dict:
    base_metadata = build_scenechange_sequence(item, perm_item, out_dir, rng)
    scene_frames = [inc.load_rgb(out_dir / "I0.png"), inc.load_rgb(out_dir / "I_0.5.png"), inc.load_rgb(out_dir / "I1.png")]
    # Reuse the augmented builder but keep the scenechanged frames as the base.
    overlay_masks_root = out_dir / "overlays_masks"
    aggregate_root = out_dir / "aggregate_masks"
    overlay_count = choose_overlay_count(rng)
    prepared: list[VimeoPreparedOverlay] = []
    for overlay_id in range(overlay_count):
        asset = sample_overlay_asset(catalog, rng)
        overlay_profile = {
            "geometry_mode": "motion" if rng.random() < 0.3 else "static",
            "geometry_submode": "translational" if rng.random() < 0.5 else "static",
            "transparent": rng.random() < 0.3,
            "temporal_mode": "none",
            "temporal_detail": {},
        }
        if overlay_profile["geometry_mode"] == "static":
            overlay_profile["geometry_submode"] = "static"
        overlay = prepare_overlay(
            asset,
            (scene_frames[0].shape[1], scene_frames[0].shape[0]),
            catalog,
            font_paths,
            overlay_profile,
            rng,
            overlay_id,
        )
        overlay.proposals = generate_proposals(scene_frames[0], overlay, rng)
        if not overlay.proposals:
            continue
        prepared.append(overlay)
    chosen = select_non_overlapping(prepared)

    out_frames = {
        "I0": scene_frames[0].copy(),
        "I_0.5": scene_frames[1].copy(),
        "I1": scene_frames[2].copy(),
        "I_0.5_copied": inc.load_rgb(out_dir / "I_0.5_copied.png"),
    }
    aggregate_masks = {name: np.zeros(scene_frames[0].shape[:2], dtype=np.uint8) for name in out_frames}
    overlays_meta = []
    grayscale_values = inc.spaced_grayscale_values(len(chosen))
    rng.shuffle(grayscale_values)
    for overlay, proposal in chosen:
        overlay.color = grayscale_values.pop()
        mask_paths: dict[str, str] = {}
        for logical_name, frame_idx in [("I0", 0), ("I_0.5", 1), ("I1", 2)]:
            variant_rgb, variant_alpha = inc.apply_temporal_transform(
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
            out_frames[logical_name], mask, clipped_box = inc.composite(
                out_frames[logical_name], variant_rgb, variant_alpha, x, y, overlay.alpha_scales[frame_idx]
            )
            mask_name = f"{overlay.overlay_id:03d}_{logical_name}.png"
            full_mask = np.zeros(scene_frames[0].shape[:2], dtype=np.uint8)
            if mask.size > 0:
                x0, y0, x1, y1 = clipped_box
                full_mask[y0:y1, x0:x1] = mask
                patch = aggregate_masks[logical_name][y0:y1, x0:x1]
                patch[mask > 0] = overlay.color
            save_gray(overlay_masks_root / mask_name, full_mask)
            mask_paths[logical_name] = f"overlays_masks/{mask_name}"
        save_gray(overlay_masks_root / f"{overlay.overlay_id:03d}_I_0.5_copied.png", np.zeros(scene_frames[0].shape[:2], dtype=np.uint8))
        mask_paths["I_0.5_copied"] = f"overlays_masks/{overlay.overlay_id:03d}_I_0.5_copied.png"
        overlays_meta.append(overlay_metadata_record(overlay, proposal, mask_paths))

    aug_frames = inc.apply_global_seq_aug([out_frames["I0"], out_frames["I_0.5"], out_frames["I1"], out_frames["I_0.5_copied"]])
    out_frames["I0"], out_frames["I_0.5"], out_frames["I1"], out_frames["I_0.5_copied"] = aug_frames
    save_rgb(out_dir / "I0.png", out_frames["I0"])
    save_rgb(out_dir / "I_0.5.png", out_frames["I_0.5"])
    save_rgb(out_dir / "I1.png", out_frames["I1"])
    save_rgb(out_dir / "I_0.5_copied.png", out_frames["I_0.5_copied"])
    for logical_name, mask_name in [("I0", "M0.png"), ("I_0.5", "M05.png"), ("I1", "M1.png"), ("I_0.5_copied", "M05_copied.png")]:
        save_gray(aggregate_root / mask_name, aggregate_masks[logical_name])

    metadata = base_metadata
    metadata["scenechange"]["preserved_go"] = True
    metadata["composite_profile"] = {
        "scope": "per_overlay",
        "overlay_count_target": overlay_count,
        "overlay_count_placed": len(chosen),
        "preserved_go": True,
    }
    metadata["tasks"] = base_task_metadata(overlays_meta)
    metadata["overlays"] = overlays_meta
    metadata["objects"] = overlays_meta
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def build_split(
    split_name: str,
    items: list[VimeoSequenceItem],
    catalog: dict[str, list[inc.CatalogEntry]],
    font_paths: list[Path],
    rng: random.Random,
) -> None:
    split_out_root = cfg.OUTPUT_ROOT / split_name
    split_out_root.mkdir(parents=True, exist_ok=True)
    perm = make_derangement(len(items), rng)
    for idx, item in enumerate(items):
        out_dir = split_out_root / item.rel
        if rng.random() < cfg.SCENECHANGE_PROB:
            perm_item = items[perm[idx]]
            if rng.random() < cfg.SCENECHANGE_PRESERVED_GO_PROB:
                build_scenechange_preserved_go_sequence(item, perm_item, out_dir, catalog, font_paths, rng)
            else:
                build_scenechange_sequence(item, perm_item, out_dir, rng)
            continue
        profile = sample_sequence_profile(rng)
        build_augmented_sequence(item, out_dir, catalog, font_paths, rng, profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Vimeo triplet augmentation dataset with detailed task metadata.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional limit for train split, for smoke testing.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional limit for val split, for smoke testing.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional limit for test split, for smoke testing.")
    parser.add_argument("--output-root", type=Path, default=cfg.OUTPUT_ROOT, help="Override output root.")
    args = parser.parse_args()

    cfg.OUTPUT_ROOT = args.output_root.resolve()
    rng = random.Random(cfg.SEED)
    train_items = list_split_items("train", cfg.VIMEO_TRAIN_LIST, args.limit_train)
    val_items = list_split_items("val", cfg.VIMEO_VAL_LIST, args.limit_val)
    test_items = list_split_items("test", cfg.VIMEO_TEST_LIST, args.limit_test)
    split_specs = [
        ("train", train_items, load_catalog(cfg.GOONS_TRAIN_SPLIT)),
        ("val", val_items, load_catalog(cfg.GOONS_VAL_SPLIT)),
        ("test", test_items, load_catalog(cfg.GOONS_TEST_SPLIT)),
    ]

    for split_name, items, catalog in split_specs:
        font_paths = inc.sample_font_paths(catalog)
        build_split(split_name, items, catalog, font_paths, rng)
    shutil.copy2(cfg.VIMEO_TRAIN_LIST, cfg.OUTPUT_ROOT / cfg.VIMEO_TRAIN_LIST.name)
    shutil.copy2(cfg.VIMEO_VAL_LIST, cfg.OUTPUT_ROOT / cfg.VIMEO_VAL_LIST.name)
    shutil.copy2(cfg.VIMEO_TEST_LIST, cfg.OUTPUT_ROOT / cfg.VIMEO_TEST_LIST.name)
    print(f"Built Vimeo augmented dataset into {cfg.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
