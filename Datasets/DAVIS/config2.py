from __future__ import annotations

import os
from pathlib import Path

from config import *  # noqa: F401,F403


# Affine-only amplified triplet augmentation for a small subset.
SEQUENCE_LENGTH = 3
SOURCE_ROOT = Path(os.environ.get("DAVIS_SOURCE_ROOT2", str(PROJECT_ROOT / "DAVIS-Triplet")))
OUTPUT_ROOT = Path(os.environ.get("DAVIS_OUTPUT_ROOT2", str(PROJECT_ROOT / "DAVIS-Triplet-Affine-Examine")))
MAX_SEQUENCES = 50

temporal_mode_probs = {
    "static": 0.3,
    "blink": 0.2,
    "affine": 0.5,
}

count_ranges = [(3, 6)]
count_ranges_prob = [1.0]

p_moving = 0.4
moving_speed_frac_range = (0.004, 0.03)

affine_two_component_prob = 0.20
affine_angle_velocity_range = (-5.0, 5.0)
affine_scale_end_range = (0.5, 2.0)
affine_squeeze_end_range = (0.8, 1.2)
affine_shear_velocity_range = (-0.07, 0.07)