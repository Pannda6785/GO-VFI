from __future__ import annotations

from pathlib import Path

import albumentations as A


PACKAGE_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PACKAGE_ROOT.parent

VIMEO_ROOT = DATASETS_ROOT / "vimeo_triplet"
VIMEO_SEQUENCE_ROOT = VIMEO_ROOT / "sequences"
VIMEO_TRAIN_LIST = VIMEO_ROOT / "tri_trainlist.txt"
VIMEO_VAL_LIST = VIMEO_ROOT / "tri_vallist.txt"
VIMEO_TEST_LIST = VIMEO_ROOT / "tri_testlist.txt"

GOONS_META_ROOT = DATASETS_ROOT / "GOoNS"
GOONS_ROOT = GOONS_META_ROOT / "GOoNS"
GOONS_TRAIN_SPLIT = GOONS_META_ROOT / "train.txt"
GOONS_VAL_SPLIT = GOONS_META_ROOT / "val.txt"
GOONS_TEST_SPLIT = GOONS_META_ROOT / "test.txt"

OUTPUT_ROOT = DATASETS_ROOT / "vimeo_triplet_augmented"
SEED = 123
SEQUENCE_LENGTH = 3

# Scenechange is a sequence-level override: if sampled, normal overlay augmentation is skipped.
SCENECHANGE_PROB = 0.02
SCENECHANGE_PRESERVED_GO_PROB = 0.3
SCENECHANGE_PERMUTE_PROB = 0.4
SCENECHANGE_DARK_RANGE = (0.75, 0.98)

# Soft target for how many overlays a sequence tries to place before subset selection.
count_ranges = [(0, 0), (1, 2), (3, 7), (8, 9), (10, 12)]
count_ranges_prob = [0.20, 0.15, 0.30, 0.20, 0.05]
GOONS_CLASS_WEIGHTS = {
    "Font": 0.05,
    "Functional": 0.25,
    "Panel": 0.25,
    "Symbol": 0.30,
    "Text": 0.15,
}

# Geometry / temporal properties are sampled independently per overlay.
GEOMETRY_MODE_PROBS = {
    "static": 0.85,
    "motion": 0.15,
}
MOTION_SUBMODE_PROBS = {
    "translational": 0.5,
    "affine": 0.3,
    "both": 0.2,
}
TRANSPARENT_PROB = 0.05
TEMPORAL_MODE_PROBS = {
    "none": 0.3,
    "change_appearance": 0.4,
    "appear_disappear": 0.3,
}
CHANGE_APPEARANCE_COMPONENT_WEIGHTS = {
    "color": 0.10,
    "visibility": 0.10,
    "textual": 0.20,
    "composite": 0.05,
}

transparent_alpha = 0.85
transparent_alpha_range = (0.65, 0.9)
# Pixels below this alpha are treated as absent for mask extraction and overlap checks.
mask_alpha_threshold = 0.1
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
    "text_to_text": 0.5,
    "none_to_text": 0.25,
    "text_to_none": 0.25,
}
appearance_change_color_scale_range = (0.75, 1.25)

# Proposal search per overlay before greedy non-overlap selection.
placement_trials_per_overlay = 28
proposal_attempts = 10
# Two overlays are rejected if exact mask overlap exceeds this ratio of the smaller mask.
max_pairwise_mask_overlap_ratio = 0.02
min_fg_alpha = 50
min_contrast_luma = 18.0
min_contrast_rgb = 24.0
# Overlay area is sampled as a fraction of the full frame area.
min_area_ratio = 0.004
max_area_ratio = 0.200

p_white_text = 0.6
font_text_length_range = (4, 12)
font_text_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%?&@"
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
            quality_range=(65, 88),
            compression_type="jpeg",
            p=0.85,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.05,
            contrast_limit=0.05,
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=8,
            val_shift_limit=5,
            p=0.55,
        ),
    ]
)
