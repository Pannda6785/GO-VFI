#!/usr/bin/env python3
"""Affine-only amplified DAVIS triplet augmentation over a 50-sequence subset."""

from __future__ import annotations

import build_davis_seq_aug as base
import config2 as cfg


base.cfg = cfg
_original_sequence_paths = base.sequence_paths


def limited_sequence_paths():
    paths = _original_sequence_paths()
    max_sequences = getattr(cfg, "MAX_SEQUENCES", None)
    if max_sequences is None:
        return paths
    return paths[:max_sequences]


base.sequence_paths = limited_sequence_paths


if __name__ == "__main__":
    base.main()
