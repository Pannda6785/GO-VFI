from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 2400
CANVAS_H = 1500
BG = "#f6f1e8"
INK = "#1f2430"
MUTED = "#5f6b7a"
LINE = "#d8cdbb"
ACCENT = "#1f6f78"
ACCENT_SOFT = "#d7ece8"
ACCENT_ALT = "#ad343e"
CARD = "#fffaf2"
CARD_ALT = "#f5eee1"
MASK_RED = "#d1495b"
MASK_BLUE = "#00798c"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_spacing: int = 6,
) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    fill: str = CARD,
    outline: str = LINE,
    title_fill: str = INK,
    body_fill: str = MUTED,
    radius: int = 26,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)
    x0, y0, x1, _ = box
    draw.text((x0 + 24, y0 + 18), title, font=title_font, fill=title_fill)
    draw_wrapped_text(
        draw,
        (x0 + 24, y0 + 58),
        body,
        font=body_font,
        fill=body_fill,
        max_width=(x1 - x0) - 48,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = ACCENT,
    width: int = 8,
    head: int = 18,
) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        points = [
            (x1, y1),
            (x1 - direction * head, y1 - head // 2),
            (x1 - direction * head, y1 + head // 2),
        ]
    else:
        direction = 1 if y1 >= y0 else -1
        points = [
            (x1, y1),
            (x1 - head // 2, y1 - direction * head),
            (x1 + head // 2, y1 - direction * head),
        ]
    draw.polygon(points, fill=color)


def connector_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    pad_x = 14
    pad_y = 8
    x = center[0] - (bbox[2] - bbox[0]) // 2 - pad_x
    y = center[1] - (bbox[3] - bbox[1]) // 2 - pad_y
    draw.rounded_rectangle(
        (x, y, x + (bbox[2] - bbox[0]) + 2 * pad_x, y + (bbox[3] - bbox[1]) + 2 * pad_y),
        radius=16,
        fill="#fffdf8",
        outline=LINE,
        width=2,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=MUTED)


def make_figure(output_path: Path) -> None:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)

    title_font = load_font(56, bold=True)
    stage_font = load_font(28, bold=True)
    box_title_font = load_font(30, bold=True)
    body_font = load_font(22, bold=False)
    small_font = load_font(18, bold=False)
    tiny_bold = load_font(18, bold=True)

    draw.text((90, 54), "GOBC Inference Pipeline and Training Objective", font=title_font, fill=INK)
    draw.text(
        (92, 122),
        "Shared frozen DINOv2 extracts patch tokens from two masked object crops; a lightweight bidirectional "
        "cross-attention comparator predicts whether the overlay pair is different.",
        font=body_font,
        fill=MUTED,
    )

    stages = [
        (90, "1. Input + Preprocess"),
        (670, "2. Shared Feature Extraction"),
        (1310, "3. Pairwise Comparator"),
        (1960, "4. Output + Loss"),
    ]
    for x, text in stages:
        draw.rounded_rectangle((x, 188, x + 340, 238), radius=24, fill=ACCENT_SOFT, outline=LINE, width=2)
        draw.text((x + 18, 198), text, font=stage_font, fill=ACCENT)

    # Input boxes
    draw_box(
        draw,
        (90, 290, 520, 500),
        "Overlay Pair Input",
        "image1 = I0 crop, image2 = I1 crop, mask1/mask2 = per-overlay masks. Label: 1=different, 0=similar.",
        title_font=box_title_font,
        body_font=body_font,
    )
    draw.rounded_rectangle((130, 378, 300, 470), radius=18, fill="#f2ede4", outline=LINE, width=2)
    draw.rounded_rectangle((320, 378, 490, 470), radius=18, fill="#f2ede4", outline=LINE, width=2)
    draw.text((150, 392), "I0 + mask1", font=tiny_bold, fill=INK)
    draw.text((340, 392), "I1 + mask2", font=tiny_bold, fill=INK)
    draw.rectangle((150, 422, 250, 450), fill="#d9e7f2", outline=None)
    draw.rectangle((170, 414, 210, 458), outline=MASK_RED, width=5)
    draw.rectangle((340, 422, 440, 450), fill="#d9e7f2", outline=None)
    draw.rectangle((360, 414, 400, 458), outline=MASK_BLUE, width=5)

    draw_box(
        draw,
        (90, 560, 520, 860),
        "Dataset and Crop Rule",
        "Drop scene-change sequences and appear_disappear overlays. Different if change_appearance is textual or "
        "composite containing textual. Compute a shared union crop from mask1 OR mask2, expand by 15%, then resize "
        "both images and masks to 224x224.",
        title_font=box_title_font,
        body_font=body_font,
        fill=CARD_ALT,
    )

    draw_box(
        draw,
        (90, 920, 520, 1220),
        "Normalization + Validity",
        "Images are normalized with DINO mean/std. Masks are projected to the 16x16 patch grid with area "
        "interpolation. A hard mask is defined as soft-mask area > 0.3, and samples with zero hard-mask patch "
        "tokens in either branch are dropped before loss.",
        title_font=box_title_font,
        body_font=body_font,
        fill=CARD_ALT,
    )

    # Shared backbone stage
    draw_box(
        draw,
        (670, 340, 1120, 650),
        "Shared Frozen DINOv2 ViT-B/14",
        "Both 224x224 crops pass through the same frozen backbone. Only patch tokens are kept; the CLS token is "
        "discarded. Patch size = 14, so each image becomes a 16x16 grid = 256 tokens.",
        title_font=box_title_font,
        body_font=body_font,
        fill="#eef4f1",
    )
    draw.rounded_rectangle((720, 480, 870, 580), radius=18, fill="#dcebe5", outline=LINE, width=2)
    draw.rounded_rectangle((920, 480, 1070, 580), radius=18, fill="#dcebe5", outline=LINE, width=2)
    draw.text((744, 500), "image1", font=tiny_bold, fill=INK)
    draw.text((947, 500), "image2", font=tiny_bold, fill=INK)
    draw.text((734, 536), "shared weights", font=small_font, fill=MUTED)
    draw.text((940, 536), "shared weights", font=small_font, fill=MUTED)

    draw_box(
        draw,
        (670, 740, 1120, 1090),
        "Patch Tokens + Patch Masks",
        "Backbone output: tokens1, tokens2 in [B,256,768]. Masks are resized to 16x16 and flattened to "
        "Per branch: soft patch masks from area projection and hard patch masks from soft_mask > 0.3, both in [B,256].",
        title_font=box_title_font,
        body_font=body_font,
        fill="#eef4f1",
    )

    # Comparator stage
    draw_box(
        draw,
        (1310, 280, 1790, 470),
        "Linear Projection + Mask Gating",
        "A shared linear layer maps DINO features from 768 to 256 dimensions. Background patch tokens are zeroed "
        "using the projected binary masks.",
        title_font=box_title_font,
        body_font=body_font,
    )

    draw_box(
        draw,
        (1310, 540, 1530, 830),
        "Cross-Attn 1→2",
        "Object-1 tokens attend to object-2 context. Block = LayerNorm, 8-head attention, residual, FFN "
        "(256→1024→256), residual.",
        title_font=box_title_font,
        body_font=body_font,
    )
    draw_box(
        draw,
        (1570, 540, 1790, 830),
        "Cross-Attn 2→1",
        "Symmetric block: object-2 tokens attend to object-1 context with the same design and dimensionality.",
        title_font=box_title_font,
        body_font=body_font,
    )

    draw_box(
        draw,
        (1310, 900, 1790, 1130),
        "Masked Mean Pool",
        "Foreground token sets are pooled with mask-weighted mean pooling to get embedding1 and embedding2, each "
        "with shape [B_valid,256].",
        title_font=box_title_font,
        body_font=body_font,
    )

    draw_box(
        draw,
        (1310, 1180, 1790, 1430),
        "Relation Vector",
        "Concatenate [z1, z2, |z1-z2|, z1*z2] to form a 1024-D relation descriptor for each valid pair.",
        title_font=box_title_font,
        body_font=body_font,
    )

    # Output stage
    draw_box(
        draw,
        (1960, 360, 2310, 620),
        "MLP Head",
        "Two-layer head: 1024→256→1 with GELU. Produces a scalar logit for each valid pair.",
        title_font=box_title_font,
        body_font=body_font,
        fill="#f7eee7",
    )
    draw_box(
        draw,
        (1960, 700, 2310, 960),
        "Sigmoid + Decision",
        "Probability of difference = sigmoid(logit). Default evaluation threshold is 0.5, though threshold tuning "
        "changes recall/precision tradeoffs substantially.",
        title_font=box_title_font,
        body_font=body_font,
        fill="#f7eee7",
    )
    draw_box(
        draw,
        (1960, 1040, 2310, 1420),
        "Training Objective",
        "Loss = BCEWithLogitsLoss(logit, label) on valid samples only. Optimizer = AdamW, lr = 1e-4, wd = 1e-4. "
        "The DINOv2 backbone stays frozen; only the projection layer, both cross-attention blocks, and the MLP head "
        "are updated.",
        title_font=box_title_font,
        body_font=body_font,
        fill="#f7eee7",
    )

    # Arrows and labels
    arrow(draw, (520, 400), (670, 400))
    connector_label(draw, (595, 365), "shared crop -> 224x224", small_font)

    arrow(draw, (1120, 490), (1310, 375))
    connector_label(draw, (1215, 398), "tokens1,tokens2: [B,256,768]", small_font)

    arrow(draw, (1120, 885), (1310, 375))
    connector_label(draw, (1218, 795), "patch masks: [B,256]", small_font)

    arrow(draw, (1550, 470), (1550, 540))
    connector_label(draw, (1665, 504), "projected dim = 256", small_font)

    arrow(draw, (1530, 690), (1570, 690))
    connector_label(draw, (1550, 650), "bidirectional", small_font)

    arrow(draw, (1550, 830), (1550, 900))
    connector_label(draw, (1658, 865), "attended tokens", small_font)

    arrow(draw, (1550, 1130), (1550, 1180))
    connector_label(draw, (1640, 1154), "z1, z2", small_font)

    arrow(draw, (1790, 1305), (1960, 490))
    connector_label(draw, (1880, 900), "relation: [B_valid,1024]", small_font)

    arrow(draw, (2135, 620), (2135, 700))
    arrow(draw, (2135, 960), (2135, 1040), color=ACCENT_ALT)
    connector_label(draw, (2240, 1000), "loss on valid pairs", small_font)

    # Side callouts
    draw_box(
        draw,
        (700, 1145, 1120, 1430),
        "Run Configuration Used",
        "image size 224, patch size 14, batch size 256 for training, batch size 128 for eval, frozen DINOv2 "
        "ViT-B/14, projection dim 256, 8 attention heads, dropout 0.0.",
        title_font=box_title_font,
        body_font=body_font,
        fill="#fffdf8",
    )
    draw_box(
        draw,
        (540, 1160, 650, 1390),
        "Mask Note",
        "Masks supervise token selection only. There is no pixel decoder, no segmentation loss, and no boundary/interior split.",
        title_font=tiny_bold,
        body_font=small_font,
        fill="#fffdf8",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="Test/GOBC_train_bs256_e10_s4092/methods_figure.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    make_figure(Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
