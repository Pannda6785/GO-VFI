from __future__ import annotations

import os
from pathlib import Path

import albumentations as A


# Dataset selection.
PROJECT_ROOT = Path(__file__).resolve().parent
GOONS_ROOT = PROJECT_ROOT / "GOoNS"
TRAIN_SPLIT = GOONS_ROOT / "train.txt"
SEQUENCE_LENGTH = 3
SEED = 123


def davis_roots(sequence_length: int) -> tuple[Path, Path]:
    if sequence_length == 3:
        return PROJECT_ROOT / "DAVIS-Triplet", PROJECT_ROOT / "DAVIS-Triplet-Aug"
    if sequence_length == 7:
        return PROJECT_ROOT / "DAVIS-Septuplet", PROJECT_ROOT / "DAVIS-Seq-Aug"
    return (
        PROJECT_ROOT / f"DAVIS-{sequence_length}-Frame",
        PROJECT_ROOT / f"DAVIS-{sequence_length}-Frame-Aug",
    )
default_source_root, default_output_root = davis_roots(SEQUENCE_LENGTH)
SOURCE_ROOT = Path(os.environ.get("DAVIS_SOURCE_ROOT", str(default_source_root)))
OUTPUT_ROOT = Path(os.environ.get("DAVIS_OUTPUT_ROOT", str(default_output_root)))

# Temporal transition behavior.
drop_middle_frame_for_appear_disappear = False

# Global augmentation and masking.
p_augment = 1.0
p_transparent = 0.05
transparent_alpha = 0.85
mask_alpha_threshold = 0.1
count_ranges = [(0, 0), (1, 2), (3,7), (8, 9), (10, 12)]
count_ranges_prob = [0.20, 0.15, 0.3, 0.20, 0.05]
overlay_class_prob = {
    "Font": 0.05,
    "Functional": 0.25,
    "Panel": 0.25,
    "Symbol": 0.30,
    "Text": 0.15,
}

# Overlay motion.
p_moving = 0.15
moving_speed_frac_range = (0.004, 0.03)

# Blink behavior.
blink_alpha_range = (0.6, 1.0)
blink_alpha_gap_bias_power = 2.0

# Affine temporal motion.
affine_two_component_prob = 0.20
affine_angle_velocity_range = (-5.0, 5.0)
affine_scale_end_range = (0.5, 2.0)
affine_squeeze_end_range = (0.8, 1.2)
affine_shear_velocity_range = (-0.07, 0.07)
affine_component_weights = {
    "rotate": 0.35,
    "scale": 0.35,
    "squeeze": 0.15,
    "shear": 0.15,
}
temporal_mode_probs = {
    "static": 0.35,
    "disappear": 0.125,
    "appear": 0.125,
    "appearance_change": 0.20,
    "blink": 0.05,
    "affine": 0.15,
}

# Appearance-change behavior.
appearance_change_nonfont_color_only_prob = 0.6
appearance_change_color_scale_range = (0.7, 1.3)
appearance_change_text_transition_probs = { # p = 1.0 - appearance_change_nonfont_color_only_prob 
    "text_to_text": 0.7,
    "none_to_text": 0.15,
    "text_to_none": 0.15,
}
appearance_change_text_and_color_prob = 0.1

# Proposal search and placement filtering.
proposal_attempts = 10
placement_trials_per_overlay = 28
max_pairwise_mask_overlap_ratio = 0.03
proposal_conflict_check_frames = "all"
min_fg_alpha = 50
min_contrast_luma = 18.0
min_contrast_rgb = 24.0
min_area_ratio = 0.004
max_area_ratio = 0.200

# Text rendering.
p_white_text = 0.5
font_text_length_range = (4, 12)
font_text_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
font_size_range = (24, 112)
font_stroke_width = (0, 3)
font_fill_palette = [
    (255, 255, 255),
    (255, 240, 120),
    (120, 255, 255),
    (255, 180, 180),
    (180, 255, 180),
]

# Per-overlay visual augmentation.
overlay_aug = A.ReplayCompose(
    [
        A.Affine(
            rotate=(-9, 9),
            shear=(-5, 5),
            fit_output=True,
            interpolation=1,
            mask_interpolation=0,
            border_mode=0,
            fill=0,
            fill_mask=0,
            p=0.6,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.10, contrast_limit=0.10, p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=1.0
                ),
                A.ChannelShuffle(p=1.0),
            ],
            p=0.35,
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=0.05),
    ]
)

# Whole-sequence visual augmentation.
global_seq_aug = A.ReplayCompose(
    [
        A.ImageCompression(
            quality_range=(45, 88),
            compression_type="jpeg",
            p=0.85,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=12,
            val_shift_limit=8,
            p=0.55,
        ),
    ]
)
