from __future__ import annotations

import os
from pathlib import Path

import albumentations as A


MINIPROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = MINIPROJECT_ROOT.parent
GOONS_ROOT = DATASETS_ROOT / "SelectedGOoNS"
TRAIN_SPLIT = GOONS_ROOT / "split.txt"

SEQUENCE_LENGTH = 3
SEED = 123
FIRST_TRIPLET_ONLY = True
REQUIRE_EXACT_CLASS_COVERAGE = True
CLASS_ORDER = ["Font", "Functional", "Panel", "Symbol", "Text"]
SEQUENCE_BUILD_ATTEMPTS = 60

SOURCE_ROOT = Path(os.environ.get("DAVIS_MINI_SOURCE_ROOT", str(DATASETS_ROOT / "DAVIS-Triplet")))
OUTPUT_ROOT = Path(os.environ.get("DAVIS_MINI_OUTPUT_ROOT", str(MINIPROJECT_ROOT / "outputs" / "undefined")))

DATASET_KEY = "undefined"
USE_SCENECHANGE_SOURCE = False
SCENECHANGE_ROOT = MINIPROJECT_ROOT / "outputs" / "scenechange"

mask_alpha_threshold = 0.1
transparent_alpha = 0.85
transparent_alpha_range = (0.65, 0.9)
moving_speed_frac_range = (0.004, 0.03)

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

appearance_change_text_transition_probs = {
    "text_to_text": 0.7,
    "none_to_text": 0.15,
    "text_to_none": 0.15,
}
appearance_change_color_scale_range = (0.7, 1.3)
visibility_change_color_scale_range = (0.65, 1.45)

placement_trials_per_overlay = 28
proposal_attempts = 10
max_pairwise_mask_overlap_ratio = 0.03
min_fg_alpha = 50
min_contrast_luma = 18.0
min_contrast_rgb = 24.0
min_area_ratio = 0.004
max_area_ratio = 0.200

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
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=12,
                    val_shift_limit=8,
                    p=1.0,
                ),
                A.ChannelShuffle(p=1.0),
            ],
            p=0.35,
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=0.05),
    ]
)

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
