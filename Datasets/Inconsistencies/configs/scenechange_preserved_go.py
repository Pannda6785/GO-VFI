from __future__ import annotations

from davis_triplet_miniproject.base_config import *  # noqa: F401,F403


DATASET_KEY = "scenechange_preserved_go"
USE_SCENECHANGE_SOURCE = True
SOURCE_ROOT = SCENECHANGE_ROOT
OUTPUT_ROOT = MINIPROJECT_ROOT / "outputs" / DATASET_KEY
