#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from build_vimeo_triplet_aug import (
    build_augmented_sequence,
    build_scenechange_preserved_go_sequence,
    build_scenechange_sequence,
    list_split_items,
    load_catalog,
    make_derangement,
    sample_sequence_profile,
    inc,
    cfg,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()

    cfg.OUTPUT_ROOT = args.output_root.resolve()
    split_root = cfg.OUTPUT_ROOT / args.split
    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.SEED)
    if args.split == "train":
        catalog = load_catalog(cfg.GOONS_TRAIN_SPLIT)
        list_path = cfg.VIMEO_TRAIN_LIST
    elif args.split == "val":
        catalog = load_catalog(cfg.GOONS_VAL_SPLIT)
        list_path = cfg.VIMEO_VAL_LIST
    else:
        catalog = load_catalog(cfg.GOONS_TEST_SPLIT)
        list_path = cfg.VIMEO_TEST_LIST
    font_paths = inc.sample_font_paths(catalog)
    items = list_split_items(args.split, list_path)
    perm = make_derangement(len(items), rng)

    successes = 0
    failures = 0
    attempts = 0
    for idx, item in enumerate(items):
        if successes >= args.count or attempts >= args.max_attempts:
            break
        attempts += 1
        out_dir = split_root / item.rel
        try:
            if rng.random() < cfg.SCENECHANGE_PROB:
                perm_item = items[perm[idx]]
                if rng.random() < cfg.SCENECHANGE_PRESERVED_GO_PROB:
                    build_scenechange_preserved_go_sequence(item, perm_item, out_dir, catalog, font_paths, rng)
                else:
                    build_scenechange_sequence(item, perm_item, out_dir, rng)
            else:
                profile = sample_sequence_profile(rng)
                build_augmented_sequence(item, out_dir, catalog, font_paths, rng, profile)
            successes += 1
        except RuntimeError:
            failures += 1
            continue

    print(
        f"Built {successes} successful {args.split} preview sequences "
        f"into {split_root} with {failures} skipped failures."
    )


if __name__ == "__main__":
    main()
