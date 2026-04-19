from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gobc-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from GOBC.models.dinov2_backbone import DINO_MEAN, DINO_STD
from GOBC.train import (
    build_criterion,
    build_dataset,
    build_loader,
    build_model,
    load_config,
    make_device,
    move_batch,
    prepare_epoch_subsets,
)


def unnormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(DINO_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(DINO_STD, dtype=image.dtype).view(3, 1, 1)
    image = image.detach().cpu() * std + mean
    image = image.clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: torch.Tensor, color: str) -> Image.Image:
    image_pil = Image.fromarray(image)
    overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    rgba = ImageColor.getrgb(color) + (120,)
    mask_np = (mask.detach().cpu().squeeze(0).numpy() > 0.5).astype(np.uint8) * 255
    overlay.paste(rgba, mask=Image.fromarray(mask_np, mode="L"))
    return Image.alpha_composite(image_pil.convert("RGBA"), overlay).convert("RGB")


def save_visual_panel(
    output_path: Path,
    image1: torch.Tensor,
    mask1: torch.Tensor,
    image2: torch.Tensor,
    mask2: torch.Tensor,
    label: float,
    score: float,
    title: str,
) -> None:
    image1_np = unnormalize_image(image1)
    image2_np = unnormalize_image(image2)
    vis1 = overlay_mask(image1_np, mask1, "#d1495b")
    vis2 = overlay_mask(image2_np, mask2, "#00798c")
    raw1 = Image.fromarray(image1_np)
    raw2 = Image.fromarray(image2_np)

    tile_w, tile_h = raw1.size
    gap = 16
    header_h = 72
    panel = Image.new("RGB", (tile_w * 4 + gap * 5, tile_h + header_h + gap * 2), (248, 242, 232))
    draw = ImageDraw.Draw(panel)
    draw.text((gap, 16), title, fill=(31, 36, 48))
    draw.text((gap, 40), f"label={label:.0f}  prob_different={score:.4f}", fill=(92, 103, 122))

    tiles = [raw1, vis1, raw2, vis2]
    captions = ["I0 crop", "I0 + mask", "I1 crop", "I1 + mask"]
    x = gap
    y = header_h
    for tile, caption in zip(tiles, captions):
        panel.paste(tile, (x, y))
        draw.text((x, y + tile_h + 6), caption, fill=(92, 103, 122))
        x += tile_w + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def keep_top_examples(examples: list[dict[str, Any]], candidate: dict[str, Any], top_k: int) -> None:
    examples.append(candidate)
    examples.sort(key=lambda item: item["priority"], reverse=True)
    del examples[top_k:]


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
    preds = (scores >= threshold).astype(np.float32)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = cm.ravel()
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics: dict[str, float | list[list[int]]] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)) if labels.size else float("nan"),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def compute_score_quantiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "p50": float(np.quantile(scores, 0.50)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
    }


def build_subgroup_rows(predictions: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        group_name = item.get(key) or "none"
        grouped.setdefault(group_name, []).append(item)

    rows: list[dict[str, Any]] = []
    for group_name, items in grouped.items():
        labels = np.asarray([row["label"] for row in items], dtype=np.float32)
        scores = np.asarray([row["score"] for row in items], dtype=np.float32)
        preds = (scores >= 0.5).astype(np.float32)
        rows.append(
            {
                "group": group_name,
                "count": int(labels.size),
                "positive_rate": float(labels.mean()) if labels.size else float("nan"),
                "accuracy": float(accuracy_score(labels, preds)) if labels.size else float("nan"),
                "mean_score": float(scores.mean()) if scores.size else float("nan"),
                "auroc": safe_auroc(labels, scores),
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["group"]))
    return rows


def plot_score_histogram(output_path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 0], bins=20, alpha=0.6, label="similar")
    plt.hist(scores[labels == 1], bins=20, alpha=0.6, label="different")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_score_boxplot(output_path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    data = []
    names = []
    if np.any(labels == 0):
        data.append(scores[labels == 0])
        names.append("similar")
    if np.any(labels == 1):
        data.append(scores[labels == 1])
        names.append("different")
    plt.figure(figsize=(5.5, 4))
    plt.boxplot(data, tick_labels=names, widths=0.5, patch_artist=True)
    plt.ylabel("Predicted probability")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(output_path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    if np.unique(labels).size < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    plt.figure(figsize=(5.5, 4))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", color="#1f6f78", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#8b97a8")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curve(output_path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    if np.unique(labels).size < 2:
        return
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    plt.figure(figsize=(5.5, 4))
    plt.plot(recall, precision, color="#ad343e", linewidth=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_calibration_curve(output_path: Path, labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(scores, edges[1:-1], right=True)
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        xs.append(float(scores[mask].mean()))
        ys.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    plt.figure(figsize=(5.5, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#8b97a8")
    if xs:
        plt.plot(xs, ys, marker="o", linewidth=2, color="#1f6f78")
        for x, y, count in zip(xs, ys, counts):
            plt.text(x, y, str(count), fontsize=8, color="#5f6b7a")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_threshold_sweep(output_path: Path, labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float | list[list[int]]]]:
    thresholds = np.linspace(0.0, 1.0, 201)
    rows = [threshold_metrics(labels, scores, float(threshold)) for threshold in thresholds]
    accuracy_values = [float(row["accuracy"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]
    precision_values = [float(row["precision"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]

    plt.figure(figsize=(6.5, 4))
    plt.plot(thresholds, accuracy_values, label="accuracy", linewidth=2)
    plt.plot(thresholds, f1_values, label="f1", linewidth=2)
    plt.plot(thresholds, precision_values, label="precision", linewidth=2)
    plt.plot(thresholds, recall_values, label="recall", linewidth=2)
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    best_accuracy = max(rows, key=lambda row: (float(row["accuracy"]), -abs(float(row["threshold"]) - 0.5)))
    best_f1 = max(rows, key=lambda row: (float(row["f1"]), -abs(float(row["threshold"]) - 0.5)))
    return {
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
    }


def write_predictions_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = make_device()
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = build_dataset(config, split=args.split)
    epoch_subsets = prepare_epoch_subsets(config, dataset, args.split, 1)
    loader = build_loader(config, dataset=dataset, split=args.split, shuffle=False, epoch_index=0, epoch_subsets=epoch_subsets)
    criterion = build_criterion(config, device)
    labels: list[float] = []
    scores: list[float] = []
    losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    mistakes: list[dict[str, Any]] = []
    false_positive_examples: list[dict[str, Any]] = []
    false_negative_examples: list[dict[str, Any]] = []
    true_positive_examples: list[dict[str, Any]] = []
    true_negative_examples: list[dict[str, Any]] = []
    uncertain_examples: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
            if output.logits.numel() == 0:
                continue
            valid_indices = output.valid_indices
            labels_batch = batch["label"].index_select(0, valid_indices)
            loss = criterion(output.logits, labels_batch)
            probs = output.prob.detach().cpu().numpy()
            labels_np = labels_batch.detach().cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)
            losses.append(loss.item())
            labels.extend(labels_np.tolist())
            scores.extend(probs.tolist())
            valid_positions = valid_indices.detach().cpu().tolist()
            source_rels = [batch["source_rel"][i] for i in valid_positions]
            object_ids = [batch["object_id"][i] for i in valid_positions]
            temporal_modes = [batch["temporal_mode"][i] for i in valid_positions]
            temporal_variants = [batch["temporal_variant"][i] for i in valid_positions]

            for idx, pred in enumerate(preds):
                raw_index = valid_positions[idx]
                row = {
                    "source_rel": source_rels[idx],
                    "object_id": object_ids[idx],
                    "temporal_mode": temporal_modes[idx],
                    "temporal_variant": temporal_variants[idx] or "none",
                    "label": float(labels_np[idx]),
                    "score": float(probs[idx]),
                    "pred": float(pred),
                    "correct": bool(pred == labels_np[idx]),
                    "confidence": float(max(probs[idx], 1.0 - probs[idx])),
                    "uncertainty": float(abs(probs[idx] - 0.5)),
                }
                prediction_rows.append(row)

                visual_candidate = {
                    **row,
                    "image1": batch["image1"][raw_index].detach().cpu(),
                    "mask1": batch["mask1"][raw_index].detach().cpu(),
                    "image2": batch["image2"][raw_index].detach().cpu(),
                    "mask2": batch["mask2"][raw_index].detach().cpu(),
                }

                keep_top_examples(
                    uncertain_examples,
                    {**visual_candidate, "priority": float(0.5 - abs(probs[idx] - 0.5))},
                    args.top_k_visuals,
                )

                if pred != labels_np[idx]:
                    mistakes.append(
                        {
                            **row,
                            "error_margin": float(abs(probs[idx] - labels_np[idx])),
                        }
                    )
                    if labels_np[idx] == 0.0:
                        keep_top_examples(false_positive_examples, {**visual_candidate, "priority": float(probs[idx])}, args.top_k_visuals)
                    else:
                        keep_top_examples(false_negative_examples, {**visual_candidate, "priority": float(1.0 - probs[idx])}, args.top_k_visuals)
                elif labels_np[idx] == 1.0:
                    keep_top_examples(true_positive_examples, {**visual_candidate, "priority": float(probs[idx])}, args.top_k_visuals)
                else:
                    keep_top_examples(true_negative_examples, {**visual_candidate, "priority": float(1.0 - probs[idx])}, args.top_k_visuals)

    labels_arr = np.asarray(labels, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_at_05 = threshold_metrics(labels_arr, scores_arr, 0.5)
    threshold_summary = plot_threshold_sweep(output_dir / "threshold_sweep.png", labels_arr, scores_arr)
    plot_score_histogram(output_dir / "score_histogram.png", labels_arr, scores_arr)
    plot_score_boxplot(output_dir / "score_boxplot.png", labels_arr, scores_arr)
    plot_roc_curve(output_dir / "roc_curve.png", labels_arr, scores_arr)
    plot_pr_curve(output_dir / "pr_curve.png", labels_arr, scores_arr)
    plot_calibration_curve(output_dir / "calibration_curve.png", labels_arr, scores_arr)

    metrics = {
        "split": args.split,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "num_samples": int(labels_arr.size),
        "positive_count": int((labels_arr == 1).sum()),
        "negative_count": int((labels_arr == 0).sum()),
        "positive_rate": float(labels_arr.mean()) if labels_arr.size else float("nan"),
        "accuracy": float(threshold_at_05["accuracy"]),
        "balanced_accuracy": float(threshold_at_05["balanced_accuracy"]),
        "precision": float(threshold_at_05["precision"]),
        "recall": float(threshold_at_05["recall"]),
        "specificity": float(threshold_at_05["specificity"]),
        "f1": float(threshold_at_05["f1"]),
        "auroc": safe_auroc(labels_arr, scores_arr),
        "average_precision": safe_average_precision(labels_arr, scores_arr),
        "brier_score": float(brier_score_loss(labels_arr, scores_arr)) if labels_arr.size else float("nan"),
        "mean_positive_score": float(scores_arr[labels_arr == 1].mean()) if np.any(labels_arr == 1) else float("nan"),
        "mean_negative_score": float(scores_arr[labels_arr == 0].mean()) if np.any(labels_arr == 0) else float("nan"),
        "score_quantiles": {
            "overall": compute_score_quantiles(scores_arr),
            "similar": compute_score_quantiles(scores_arr[labels_arr == 0]),
            "different": compute_score_quantiles(scores_arr[labels_arr == 1]),
        },
        "threshold_at_0_5": threshold_at_05,
        "best_thresholds": threshold_summary,
        "confusion_matrix": threshold_at_05["confusion_matrix"],
        "mistakes": sorted(mistakes, key=lambda item: item["error_margin"], reverse=True)[: args.top_k_mistakes],
        "subgroup_metrics": {
            "temporal_mode": build_subgroup_rows(prediction_rows, "temporal_mode"),
            "temporal_variant": build_subgroup_rows(prediction_rows, "temporal_variant"),
        },
        "artifacts": {
            "score_histogram": "score_histogram.png",
            "score_boxplot": "score_boxplot.png",
            "roc_curve": "roc_curve.png",
            "pr_curve": "pr_curve.png",
            "calibration_curve": "calibration_curve.png",
            "threshold_sweep": "threshold_sweep.png",
            "predictions": "predictions.jsonl",
        },
        "false_positive_visuals": [],
        "false_negative_visuals": [],
        "true_positive_visuals": [],
        "true_negative_visuals": [],
        "uncertain_visuals": [],
    }

    def save_example_group(folder_name: str, examples: list[dict[str, Any]], metrics_key: str) -> None:
        saved_items: list[dict[str, Any]] = []
        for rank, example in enumerate(examples, start=1):
            filename = f"{rank:02d}_{example['source_rel'].replace('/', '_')}_{example['object_id']}.png"
            rel_path = Path("visuals") / folder_name / filename
            save_visual_panel(
                output_dir / rel_path,
                example["image1"],
                example["mask1"],
                example["image2"],
                example["mask2"],
                example["label"],
                example["score"],
                title=f"{example['source_rel']}  object {example['object_id']}  {example['temporal_variant']}",
            )
            saved_items.append(
                {
                    "source_rel": example["source_rel"],
                    "object_id": example["object_id"],
                    "temporal_mode": example["temporal_mode"],
                    "temporal_variant": example["temporal_variant"],
                    "label": example["label"],
                    "score": example["score"],
                    "image": rel_path.as_posix(),
                }
            )
        metrics[metrics_key] = saved_items

    save_example_group("false_positives", false_positive_examples, "false_positive_visuals")
    save_example_group("false_negatives", false_negative_examples, "false_negative_visuals")
    save_example_group("true_positives", true_positive_examples, "true_positive_visuals")
    save_example_group("true_negatives", true_negative_examples, "true_negative_visuals")
    save_example_group("uncertain", uncertain_examples, "uncertain_visuals")

    write_predictions_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="GOBC/configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="GOBC/outputs/eval")
    parser.add_argument("--top-k-mistakes", type=int, default=25)
    parser.add_argument("--top-k-visuals", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
