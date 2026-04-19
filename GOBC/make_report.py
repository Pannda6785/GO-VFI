from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gobc-matplotlib")

import matplotlib.pyplot as plt
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def render_table(rows: list[tuple[str, str]]) -> str:
    table_rows = "\n".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f"<table>{table_rows}</table>"


def render_plot_cards(plots: list[tuple[str, str, str]]) -> str:
    cards = []
    for title, description, rel_path in plots:
        cards.append(
            "<article class='plot-card'>"
            f"<h3>{title}</h3>"
            f"<p>{description}</p>"
            f"<img src=\"{rel_path}\" alt=\"{title}\">"
            "</article>"
        )
    return "<div class='plot-grid'>" + "\n".join(cards) + "</div>"


def render_mistakes(mistakes: list[dict[str, Any]]) -> str:
    rows = []
    for item in mistakes:
        rows.append(
            "<tr>"
            f"<td>{item['source_rel']}</td>"
            f"<td>{item['object_id']}</td>"
            f"<td>{item.get('temporal_variant', 'none')}</td>"
            f"<td>{format_metric(item['label'])}</td>"
            f"<td>{format_metric(item['score'])}</td>"
            f"<td>{format_metric(item.get('error_margin'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='6'>No mistakes recorded.</td></tr>")
    return (
        "<table>"
        "<tr><th>Sequence</th><th>Object</th><th>Variant</th><th>Label</th><th>Score</th><th>Error Margin</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def render_gallery(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p>No visuals were saved for this section.</p>"
    cards = []
    for item in items:
        cards.append(
            "<figure class='viz-card'>"
            f"<img src=\"{item['image']}\" alt=\"{item['source_rel']} {item['object_id']}\">"
            "<figcaption>"
            f"<strong>{item['source_rel']}</strong><br>"
            f"object {item['object_id']}<br>"
            f"mode={item.get('temporal_mode', 'n/a')} variant={item.get('temporal_variant', 'n/a')}<br>"
            f"label={format_metric(item['label'])} score={format_metric(item['score'])}"
            "</figcaption>"
            "</figure>"
        )
    return "<div class='viz-grid'>" + "\n".join(cards) + "</div>"


def render_subgroup_table(rows: list[dict[str, Any]], title: str) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['group']}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{format_metric(row['positive_rate'])}</td>"
            f"<td>{format_metric(row['accuracy'])}</td>"
            f"<td>{format_metric(row['mean_score'])}</td>"
            f"<td>{format_metric(row['auroc'])}</td>"
            "</tr>"
        )
    if not body:
        body.append("<tr><td colspan='6'>No subgroup rows available.</td></tr>")
    return (
        "<section class='card'>"
        f"<h2>{title}</h2>"
        "<table>"
        "<tr><th>Group</th><th>Count</th><th>Positive Rate</th><th>Accuracy</th><th>Mean Score</th><th>AUROC</th></tr>"
        + "\n".join(body)
        + "</table>"
        "</section>"
    )


def maybe_plot_epoch_trends(epoch_rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if not epoch_rows:
        return None

    epochs = [int(row["epoch"]) for row in epoch_rows]
    train_loss = [float(row["train_loss"]) if row.get("train_loss") is not None else float("nan") for row in epoch_rows]
    val_loss = [float(row["loss"]) if row.get("loss") is not None else float("nan") for row in epoch_rows]
    accuracy = [float(row["accuracy"]) if row.get("accuracy") is not None else float("nan") for row in epoch_rows]
    auroc = [float(row["auroc"]) if row.get("auroc") is not None else float("nan") for row in epoch_rows]
    precision = [float(row["precision"]) if row.get("precision") is not None else float("nan") for row in epoch_rows]
    recall = [float(row["recall"]) if row.get("recall") is not None else float("nan") for row in epoch_rows]
    pos_score = [float(row["mean_positive_score"]) if row.get("mean_positive_score") is not None else float("nan") for row in epoch_rows]
    neg_score = [float(row["mean_negative_score"]) if row.get("mean_negative_score") is not None else float("nan") for row in epoch_rows]

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.flatten()
    plot_specs = [
        ("Train Loss", train_loss),
        ("Val Loss", val_loss),
        ("Val Accuracy", accuracy),
        ("Val AUROC", auroc),
        ("Precision / Recall", None),
        ("Mean Score by Class", None),
    ]
    for ax, (title, values) in zip(axes, plot_specs):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        if values is not None:
            ax.plot(epochs, values, marker="o", linewidth=2, color="#1f6f78")
        elif title == "Precision / Recall":
            ax.plot(epochs, precision, marker="o", linewidth=2, label="precision", color="#ad343e")
            ax.plot(epochs, recall, marker="o", linewidth=2, label="recall", color="#1f6f78")
            ax.legend()
        else:
            ax.plot(epochs, pos_score, marker="o", linewidth=2, label="mean positive score", color="#ad343e")
            ax.plot(epochs, neg_score, marker="o", linewidth=2, label="mean negative score", color="#1f6f78")
            ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build_report(
    config: dict[str, Any],
    metrics: dict[str, Any],
    plot_items: list[tuple[str, str, str]],
    epoch_count: int,
) -> str:
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    best_accuracy = metrics.get("best_thresholds", {}).get("best_accuracy", {})
    best_f1 = metrics.get("best_thresholds", {}).get("best_f1", {})
    summary_rows = [
        ("Generated", datetime.now().isoformat(timespec="seconds")),
        ("Split", str(metrics.get("split"))),
        ("Loss", format_metric(metrics.get("loss"))),
        ("Precision", format_metric(metrics.get("precision"))),
        ("Recall", format_metric(metrics.get("recall"))),
        ("Specificity", format_metric(metrics.get("specificity"))),
        ("Balanced Accuracy", format_metric(metrics.get("balanced_accuracy"))),
        ("Brier Score", format_metric(metrics.get("brier_score"))),
    ]
    run_rows = [
        ("Dataset root", str(config["data"]["dataset_root"])),
        ("Image size", str(config["data"]["image_size"])),
        ("Patch size", str(config["data"].get("patch_size", 14))),
        ("Batch size", str(config["train"]["batch_size"])),
        ("Epoch budget", str(config["train"]["epochs"])),
        ("Epochs logged", str(epoch_count)),
        ("Train subset / epoch", str(config["data"].get("max_train_samples", "full"))),
        ("Val subset", str(config["data"].get("max_val_samples", "full"))),
        ("Backbone", f"{config['model']['backbone_source']}:{config['model']['backbone_name']}"),
    ]
    balance_rows = [
        ("Samples", str(metrics.get("num_samples"))),
        ("Different count", str(metrics.get("positive_count"))),
        ("Similar count", str(metrics.get("negative_count"))),
        ("Different rate", format_metric(metrics.get("positive_rate"))),
        ("Mean different score", format_metric(metrics.get("mean_positive_score"))),
        ("Mean similar score", format_metric(metrics.get("mean_negative_score"))),
    ]
    threshold_rows = [
        ("Threshold @ 0.5 accuracy", format_metric(metrics.get("accuracy"))),
        ("Best accuracy threshold", format_metric(best_accuracy.get("threshold"))),
        ("Best accuracy", format_metric(best_accuracy.get("accuracy"))),
        ("Best F1 threshold", format_metric(best_f1.get("threshold"))),
        ("Best F1", format_metric(best_f1.get("f1"))),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GOBC Run Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffaf2;
      --ink: #1f2430;
      --muted: #5f6b7a;
      --line: #d8cdbb;
      --accent: #1f6f78;
      --accent-soft: #d7ece8;
      --bad: #ad343e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ec 0, #fff8ec 18%, transparent 19%),
        linear-gradient(135deg, #efe6d4, var(--bg));
      min-height: 100vh;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.1;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 16px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(31, 36, 48, 0.08);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: linear-gradient(180deg, #fffdf8, #f6efe4);
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 26px;
      font-weight: 700;
      color: var(--accent);
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 10px 8px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      width: 34%;
    }}
    .cm {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-top: 12px;
    }}
    .cm div {{
      border-radius: 16px;
      padding: 18px;
      background: var(--accent-soft);
      border: 1px solid var(--line);
    }}
    .cm strong {{
      display: block;
      font-size: 28px;
      color: var(--accent);
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }}
    .plot-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: linear-gradient(180deg, #fffef9, #f7efe3);
    }}
    img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: white;
    }}
    .viz-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .viz-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: linear-gradient(180deg, #fffef9, #f7efe3);
    }}
    .viz-card img {{
      display: block;
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
    }}
    .viz-card figcaption {{
      padding: 12px 14px 14px;
      font-size: 14px;
      color: var(--muted);
    }}
    @media (max-width: 960px) {{
      .hero, .section-grid, .metric-grid, .plot-grid, .viz-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="card">
        <h1>GOBC Run Report</h1>
        <p>Validation analytics for the overlay-pair classifier using the per-overlay textual-change label rule, union crops, and frozen DINOv2 features.</p>
        <div class="metric-grid">
          <div class="metric"><span class="label">AUROC</span><span class="value">{format_metric(metrics.get("auroc"))}</span></div>
          <div class="metric"><span class="label">AP</span><span class="value">{format_metric(metrics.get("average_precision"))}</span></div>
          <div class="metric"><span class="label">Accuracy</span><span class="value">{format_metric(metrics.get("accuracy"))}</span></div>
          <div class="metric"><span class="label">F1</span><span class="value">{format_metric(metrics.get("f1"))}</span></div>
          <div class="metric"><span class="label">Recall</span><span class="value">{format_metric(metrics.get("recall"))}</span></div>
          <div class="metric"><span class="label">Specificity</span><span class="value">{format_metric(metrics.get("specificity"))}</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Summary</h2>
        {render_table(summary_rows)}
      </div>
    </section>

    <section class="section-grid">
      <div class="card">
        <h2>Run Config</h2>
        {render_table(run_rows)}
      </div>
      <div class="card">
        <h2>Class Balance</h2>
        {render_table(balance_rows)}
      </div>
      <div class="card">
        <h2>Threshold Tuning</h2>
        {render_table(threshold_rows)}
      </div>
      <div class="card">
        <h2>Confusion Matrix</h2>
        <p>Rows are true labels. Columns are predictions at threshold 0.5.</p>
        <div class="cm">
          <div><span>True Similar / Pred Similar</span><strong>{cm[0][0]}</strong></div>
          <div><span>True Similar / Pred Different</span><strong>{cm[0][1]}</strong></div>
          <div><span>True Different / Pred Similar</span><strong>{cm[1][0]}</strong></div>
          <div><span>True Different / Pred Different</span><strong>{cm[1][1]}</strong></div>
        </div>
      </div>
    </section>

    <section class="card" style="margin-bottom: 24px;">
      <h2>Plots</h2>
      <p>Global validation behavior, calibration, threshold sensitivity, and epoch-over-epoch movement.</p>
      {render_plot_cards(plot_items)}
    </section>

    <section class="section-grid">
      {render_subgroup_table(metrics.get("subgroup_metrics", {}).get("temporal_mode", []), "Breakdown by Temporal Mode")}
      {render_subgroup_table(metrics.get("subgroup_metrics", {}).get("temporal_variant", []), "Breakdown by Temporal Variant")}
    </section>

    <section class="card">
      <h2>Highest-Confidence Mistakes</h2>
      <p>The rows below show the strongest validation errors after sorting by how far the score landed from the true label.</p>
      {render_mistakes(metrics.get("mistakes", []))}
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>False Positives</h2>
      <p>Most confident cases where the model predicted different for a similar pair.</p>
      {render_gallery(metrics.get("false_positive_visuals", []))}
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>False Negatives</h2>
      <p>Most confident misses where the model predicted similar for a different pair.</p>
      {render_gallery(metrics.get("false_negative_visuals", []))}
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>True Positives</h2>
      <p>Most confident correct different predictions.</p>
      {render_gallery(metrics.get("true_positive_visuals", []))}
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>True Negatives</h2>
      <p>Most confident correct similar predictions.</p>
      {render_gallery(metrics.get("true_negative_visuals", []))}
    </section>

    <section class="card" style="margin-top: 24px;">
      <h2>Most Uncertain Cases</h2>
      <p>Examples with probabilities closest to 0.5, regardless of correctness.</p>
      {render_gallery(metrics.get("uncertain_visuals", []))}
    </section>
  </main>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch-metrics", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    metrics_path = Path(args.metrics)
    output_path = Path(args.output)
    epoch_metrics_path = Path(args.epoch_metrics) if args.epoch_metrics else output_path.parent / "epoch_metrics.jsonl"

    config = load_yaml(config_path)
    metrics = load_json(metrics_path)
    epoch_rows = load_jsonl(epoch_metrics_path)

    artifacts = metrics.get("artifacts", {})
    plot_items: list[tuple[str, str, str]] = []

    def add_plot(title: str, description: str, key: str) -> None:
        rel = artifacts.get(key)
        if not rel:
            return
        rel_path = (metrics_path.parent / rel).relative_to(output_path.parent).as_posix()
        plot_items.append((title, description, rel_path))

    epoch_plot_path = maybe_plot_epoch_trends(epoch_rows, output_path.parent / "epoch_trends.png")
    if epoch_plot_path is not None:
        plot_items.append(
            (
                "Epoch Trends",
                "Train loss, validation loss, accuracy, AUROC, precision/recall, and class score separation across epochs.",
                epoch_plot_path.relative_to(output_path.parent).as_posix(),
            )
        )

    add_plot("Score Histogram", "Distribution of predicted different-probabilities for similar vs different pairs.", "score_histogram")
    add_plot("Score Boxplot", "Spread of predicted scores by ground-truth class.", "score_boxplot")
    add_plot("ROC Curve", "Threshold-free tradeoff between false positives and true positives.", "roc_curve")
    add_plot("PR Curve", "Precision-recall tradeoff on the validation subset.", "pr_curve")
    add_plot("Calibration Curve", "Observed positive rate versus predicted probability by score bin.", "calibration_curve")
    add_plot("Threshold Sweep", "Accuracy, F1, precision, and recall across decision thresholds.", "threshold_sweep")

    for key in ("false_positive_visuals", "false_negative_visuals", "true_positive_visuals", "true_negative_visuals", "uncertain_visuals"):
        for item in metrics.get(key, []):
            item["image"] = (metrics_path.parent / item["image"]).relative_to(output_path.parent).as_posix()

    html = build_report(config, metrics, plot_items, len(epoch_rows))
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
