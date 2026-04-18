#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


CELL_W = 220
CELL_H = 124
INNER_PAD = 8
HEADER_H = 20
SEQ_GAP = 18
GRID_GAP = 28
BG = (245, 245, 240)
FG = (25, 25, 25)
MUTED = (110, 110, 110)
MASK_BG = (255, 255, 255)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_image(image: Image.Image, size: tuple[int, int], bg: tuple[int, int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def colorize_mask(path: Path) -> Image.Image:
    gray = Image.open(path).convert("L")
    rgb = Image.new("RGB", gray.size, MASK_BG)
    px = gray.load()
    out = rgb.load()
    colors = [
        (0, 0, 0),
        (220, 70, 70),
        (70, 150, 220),
        (80, 170, 110),
        (220, 180, 60),
        (180, 90, 200),
        (40, 160, 160),
    ]
    for y in range(gray.size[1]):
        for x in range(gray.size[0]):
            value = px[x, y]
            if value == 0:
                continue
            idx = max(1, min(len(colors) - 1, value // 32))
            out[x, y] = colors[idx]
    return rgb


def sequence_panel(seq_dir: Path) -> Image.Image:
    title = str(seq_dir.relative_to(seq_dir.parents[1]))
    labels = ["I0", "I0.5", "I0.5_copied", "I1", "M0", "M05", "M05_copied", "M1"]
    image_paths = [
        seq_dir / "I0.png",
        seq_dir / "I_0.5.png",
        seq_dir / "I_0.5_copied.png",
        seq_dir / "I1.png",
        seq_dir / "aggregate_masks" / "M0.png",
        seq_dir / "aggregate_masks" / "M05.png",
        seq_dir / "aggregate_masks" / "M05_copied.png",
        seq_dir / "aggregate_masks" / "M1.png",
    ]
    panel_w = CELL_W * 4 + INNER_PAD * 5
    panel_h = HEADER_H + CELL_H * 2 + INNER_PAD * 5 + 16
    panel = Image.new("RGB", (panel_w, panel_h), BG)
    draw = ImageDraw.Draw(panel)
    draw.text((INNER_PAD, INNER_PAD), title, fill=FG)
    for idx, (label, path) in enumerate(zip(labels, image_paths)):
        col = idx % 4
        row = idx // 3
        row = idx // 4
        x = INNER_PAD + col * (CELL_W + INNER_PAD)
        y = HEADER_H + INNER_PAD * 2 + row * (CELL_H + 18 + INNER_PAD)
        if row == 1:
            image = colorize_mask(path)
            fitted = fit_image(image, (CELL_W, CELL_H), MASK_BG)
        else:
            image = load_rgb(path)
            fitted = fit_image(image, (CELL_W, CELL_H), (250, 250, 250))
        panel.paste(fitted, (x, y))
        draw.rectangle((x, y, x + CELL_W, y + CELL_H), outline=(210, 210, 210), width=1)
        draw.text((x, y + CELL_H + 2), label, fill=MUTED)
    return panel


def is_valid_sequence_dir(seq_dir: Path) -> bool:
    required = [
        seq_dir / "I0.png",
        seq_dir / "I_0.5.png",
        seq_dir / "I_0.5_copied.png",
        seq_dir / "I1.png",
        seq_dir / "aggregate_masks" / "M0.png",
        seq_dir / "aggregate_masks" / "M05.png",
        seq_dir / "aggregate_masks" / "M05_copied.png",
        seq_dir / "aggregate_masks" / "M1.png",
    ]
    return all(path.exists() for path in required)


def make_sheet(sequence_dirs: list[Path], output_path: Path, columns: int) -> None:
    if not sequence_dirs:
        raise SystemExit("No sequence directories found.")
    panels = [sequence_panel(path) for path in sequence_dirs]
    panel_w, panel_h = panels[0].size
    cols = max(1, columns)
    rows = math.ceil(len(panels) / cols)
    sheet_w = cols * panel_w + (cols + 1) * GRID_GAP
    sheet_h = rows * panel_h + (rows + 1) * GRID_GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    for idx, panel in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x = GRID_GAP + col * (panel_w + GRID_GAP)
        y = GRID_GAP + row * (panel_h + GRID_GAP)
        sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_root = args.root / args.split
    sequence_dirs = [
        path
        for path in sorted(split_root.glob("*/*"))
        if path.is_dir() and is_valid_sequence_dir(path)
    ][: args.limit]
    make_sheet(sequence_dirs, args.output, args.columns)
    print(f"Wrote preview sheet to {args.output}")


if __name__ == "__main__":
    main()
