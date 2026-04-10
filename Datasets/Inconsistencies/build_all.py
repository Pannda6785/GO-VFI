#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = ROOT / "build_dataset.py"

CONFIG_ORDER = [
    ROOT / "configs" / "consistent_static.py",
    ROOT / "configs" / "consistent_static_transparent.py",
    ROOT / "configs" / "consistent_static_motion.py",
    ROOT / "configs" / "consistent_static_motion_transparent.py",
    ROOT / "configs" / "scenechange.py",
    ROOT / "configs" / "scenechange_preserved_go.py",
    ROOT / "configs" / "object_disappear.py",
    ROOT / "configs" / "object_texture_change.py",
    ROOT / "configs" / "object_color_change.py",
    ROOT / "configs" / "object_visibility_change.py",
    ROOT / "configs" / "object_shape_change.py",
]


def main() -> None:
    for config_path in CONFIG_ORDER:
        subprocess.run([sys.executable, str(BUILD_SCRIPT), "--config", str(config_path)], check=True)


if __name__ == "__main__":
    main()
