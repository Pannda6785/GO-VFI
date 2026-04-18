from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path


KIND_ORDER = ["i05_cp", "pred_patch", "pred", "i05_gt", "M", "E", "T"]
KIND_LABELS = {
    "i05_cp": "Interpolated Copied Input",
    "pred_patch": "Predicted Patch",
    "pred": "Refined Output",
    "i05_gt": "Target Copied GT",
    "M": "M",
    "E": "E",
    "T": "T",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HTML gallery for GORe validation outputs.")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("GORe/exp_8m_270_30"),
        help="Experiment directory containing val_outputs and loss_plot.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <exp-dir>/gallery.html",
    )
    return parser.parse_args()


def group_epoch_files(epoch_dir: Path) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for image_path in sorted(epoch_dir.glob("*.png")):
        name = image_path.stem
        for kind in KIND_ORDER:
            suffix = f"_{kind}"
            if name.endswith(suffix):
                sample_id = name[: -len(suffix)]
                groups[sample_id][kind] = image_path.name
                break
    return dict(groups)


def render_epoch(epoch_dir: Path, exp_dir: Path) -> str:
    groups = group_epoch_files(epoch_dir)
    relative_epoch = epoch_dir.relative_to(exp_dir)
    cards = []
    for sample_id, files in sorted(groups.items()):
        tiles = []
        for kind in KIND_ORDER:
            file_name = files.get(kind)
            if not file_name:
                continue
            src = html.escape(str(relative_epoch / file_name))
            label = html.escape(KIND_LABELS[kind])
            tiles.append(
                f"""
                <figure class="tile">
                  <img src="{src}" loading="lazy" />
                  <figcaption>{label}</figcaption>
                </figure>
                """
            )
        cards.append(
            f"""
            <section class="sample">
              <h3>{html.escape(sample_id)}</h3>
              <div class="grid">
                {''.join(tiles)}
              </div>
            </section>
            """
        )
    return f"""
    <section class="epoch" id="{html.escape(epoch_dir.name)}">
      <h2>{html.escape(epoch_dir.name.replace('_', ' ').title())}</h2>
      {''.join(cards)}
    </section>
    """


def build_html(exp_dir: Path) -> str:
    epoch_dirs = sorted(path for path in (exp_dir / "val_outputs").iterdir() if path.is_dir())
    nav_links = "".join(
        f'<a href="#{html.escape(epoch_dir.name)}">{html.escape(epoch_dir.name)}</a>'
        for epoch_dir in epoch_dirs
    )
    loss_plot = exp_dir / "loss_plot.png"
    loss_plot_block = ""
    if loss_plot.exists():
        loss_plot_block = f"""
        <section class="hero">
          <h1>GORe Validation Gallery</h1>
          <p>Interpolated copied input, predicted patch, refined output, target, and masks for every validation sample after each epoch.</p>
          <img class="plot" src="{html.escape(loss_plot.name)}" alt="Loss plot" />
        </section>
        """

    epoch_sections = "".join(render_epoch(epoch_dir, exp_dir) for epoch_dir in epoch_dirs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GORe Validation Gallery</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffaf2;
      --ink: #1d1a16;
      --muted: #6f6559;
      --line: #d6ccbd;
      --accent: #9a5c2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #efe6d7 0%, var(--bg) 40%, #ece7de 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .page {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }}
    .hero, .epoch {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 20px 40px rgba(78, 59, 36, 0.08);
    }}
    .hero h1, .epoch h2, .sample h3 {{
      margin: 0 0 12px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0 0 16px;
      color: var(--muted);
      max-width: 900px;
    }}
    .plot {{
      display: block;
      width: min(900px, 100%);
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 0 18px;
      backdrop-filter: blur(12px);
    }}
    .nav a {{
      text-decoration: none;
      color: white;
      background: var(--accent);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 14px;
    }}
    .sample {{
      padding-top: 12px;
      border-top: 1px solid var(--line);
      margin-top: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .tile {{
      margin: 0;
      background: white;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    .tile img {{
      width: 100%;
      display: block;
      background: #ddd;
    }}
    .tile figcaption {{
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      .page {{ width: min(100vw - 20px, 100%); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    {loss_plot_block}
    <nav class="nav">{nav_links}</nav>
    {epoch_sections}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    output_path = args.output.resolve() if args.output else (exp_dir / "gallery.html")
    html_text = build_html(exp_dir)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
