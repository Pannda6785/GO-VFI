from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from torch.utils.tensorboard import SummaryWriter

from GOBC.data.pair_dataset import VimeoOverlayPairDataset
from GOBC.models.cross_attention_matcher import PairwiseDifferenceModel


def debug_log(payload: dict[str, Any]) -> None:
    if os.environ.get("GOBC_DEBUG_MPS") != "1":
        return
    print(json.dumps({"debug": True, **payload}), flush=True)


def make_device() -> torch.device:
    requested = os.environ.get("GOBC_DEVICE")
    if requested:
        requested = requested.lower()
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("GOBC_DEVICE=cuda requested but CUDA is unavailable.")
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("GOBC_DEVICE=mps requested but MPS is unavailable.")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise RuntimeError(f"Unsupported GOBC_DEVICE override: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config: dict[str, Any], split: str) -> VimeoOverlayPairDataset:
    patch_size = int(config["data"].get("patch_size", 14))
    min_patch_tokens = int(config["data"].get("min_patch_tokens", 1))
    mask_threshold = float(config["data"].get("mask_threshold", 0.3))
    swap_pair_probability = float(config["data"].get("swap_pair_probability", 0.5))
    return VimeoOverlayPairDataset(
        dataset_root=config["data"]["dataset_root"],
        split=split,
        image_size=config["data"]["image_size"],
        crop_margin=config["data"]["crop_margin"],
        normalize=True,
        patch_size=patch_size,
        min_patch_tokens=min_patch_tokens,
        mask_threshold=mask_threshold,
        swap_pair_probability=swap_pair_probability,
    )


def make_loader(config: dict[str, Any], dataset: Dataset[Any], shuffle: bool) -> DataLoader:
    split = getattr(dataset, "split", None)
    if split is None and isinstance(dataset, Subset):
        split = getattr(dataset.dataset, "split", None)
    batch_size = config["train"]["batch_size"] if split == "train" else config["eval"].get("batch_size", config["train"]["batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def _collect_valid_indices(
    dataset: VimeoOverlayPairDataset,
    count: int,
    seed: int,
    progress_label: str | None = None,
    allow_partial: bool = False,
) -> list[int]:
    if count <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    candidate_indices = torch.randperm(len(dataset), generator=generator).tolist()
    indices: list[int] = []
    for index in candidate_indices:
        if dataset.is_valid_index(index):
            indices.append(index)
            if progress_label and (len(indices) % 2048 == 0 or len(indices) == count):
                print(
                    json.dumps({"phase": "subset_collection", "split": progress_label, "collected": len(indices), "target": count}),
                    flush=True,
                )
        if len(indices) >= count:
            break
    if len(indices) < count and not allow_partial:
        raise RuntimeError(
            f"Requested {count} valid samples from split {dataset.split}, but only collected {len(indices)}."
        )
    return indices


def prepare_epoch_subsets(
    config: dict[str, Any],
    dataset: VimeoOverlayPairDataset,
    split: str,
    num_epochs: int,
) -> list[list[int]] | None:
    split_limit_key = f"max_{split}_samples"
    split_limit = config["data"].get(split_limit_key)
    if split_limit is None:
        return None

    split_limit = min(int(split_limit), len(dataset))
    subset_seed = int(config["data"].get("subset_seed", 0))

    if split == "train" and bool(config["data"].get("distinct_train_subset_per_epoch", False)):
        required = split_limit * max(num_epochs, 1)
        valid_indices = _collect_valid_indices(
            dataset,
            min(required, len(dataset)),
            subset_seed,
            progress_label=split,
            allow_partial=True,
        )
        if len(valid_indices) < required:
            print(
                json.dumps(
                    {
                        "phase": "subset_collection_shortfall",
                        "split": split,
                        "requested": required,
                        "collected": len(valid_indices),
                        "epochs": num_epochs,
                    }
                ),
                flush=True,
            )
            if not valid_indices:
                raise RuntimeError(f"No valid samples available for split {split}.")
            return [list(chunk) for chunk in torch.tensor_split(torch.tensor(valid_indices), num_epochs)]
        return [valid_indices[i * split_limit : (i + 1) * split_limit] for i in range(num_epochs)]

    valid_indices = _collect_valid_indices(dataset, split_limit, subset_seed, progress_label=split)
    return [valid_indices for _ in range(max(num_epochs, 1))]


def build_loader(
    config: dict[str, Any],
    dataset: VimeoOverlayPairDataset,
    split: str,
    shuffle: bool,
    epoch_index: int = 0,
    epoch_subsets: list[list[int]] | None = None,
) -> DataLoader:
    subset_dataset: Dataset[Any] = dataset
    if epoch_subsets is not None:
        if epoch_index >= len(epoch_subsets):
            raise IndexError(f"Epoch index {epoch_index} is out of range for {split} subsets.")
        subset_dataset = Subset(dataset, epoch_subsets[epoch_index])
    return make_loader(config, subset_dataset, shuffle=shuffle)


def build_model(config: dict[str, Any]) -> PairwiseDifferenceModel:
    model_cfg = config["model"]
    return PairwiseDifferenceModel(
        backbone_source=model_cfg["backbone_source"],
        backbone_name=model_cfg["backbone_name"],
        backbone_hf_name=model_cfg.get("backbone_hf_name", "facebook/dinov2-base"),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        proj_dim=model_cfg["proj_dim"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        mask_threshold=config["data"]["mask_threshold"],
        return_debug_tensors=model_cfg.get("return_debug_tensors", False),
    )


def resolve_pos_weight_value(
    config: dict[str, Any],
    train_dataset: VimeoOverlayPairDataset | None = None,
) -> float:
    if "resolved_pos_weight" in config["train"]:
        return float(config["train"]["resolved_pos_weight"])

    raw_value = config["train"].get("pos_weight", 1.0)
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        config["train"]["resolved_pos_weight"] = value
        return value

    raw_text = str(raw_value).strip().lower()
    if raw_text != "auto":
        raise RuntimeError(f"Unsupported train.pos_weight setting: {raw_value!r}")

    if train_dataset is None:
        raise RuntimeError("train.pos_weight=auto requires the full filtered train dataset.")

    positive = 0
    negative = 0
    checked = 0
    for index, sample in enumerate(train_dataset.samples):
        checked += 1
        if train_dataset.is_valid_index(index):
            if int(sample.label) == 1:
                positive += 1
            else:
                negative += 1
        if checked % 8192 == 0 or checked == len(train_dataset.samples):
            print(
                json.dumps(
                    {
                        "phase": "pos_weight_resolution",
                        "checked": checked,
                        "total": len(train_dataset.samples),
                        "positive": positive,
                        "negative": negative,
                    }
                ),
                flush=True,
            )

    if positive <= 0:
        raise RuntimeError("Resolved train dataset has no positive samples; cannot compute automatic pos_weight.")
    if negative <= 0:
        raise RuntimeError("Resolved train dataset has no negative samples; cannot compute automatic pos_weight.")

    value = float(negative / positive)
    config["train"]["resolved_pos_weight"] = value
    return value


def build_criterion(
    config: dict[str, Any],
    device: torch.device,
    train_dataset: VimeoOverlayPairDataset | None = None,
) -> nn.Module:
    pos_weight_value = resolve_pos_weight_value(config, train_dataset=train_dataset)
    if pos_weight_value == 1.0:
        return nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def move_batch(batch: dict[str, Any], device: torch.device, include_label: bool = True) -> dict[str, Any]:
    moved = dict(batch)
    keys = ["image1", "image2", "mask1", "mask2"]
    if include_label:
        keys.append("label")
    for key in keys:
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def filter_finite_batch_tensors(
    logits: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(logits) & torch.isfinite(probs) & torch.isfinite(labels)
    return logits[finite], probs[finite], labels[finite]


def sanitize_binary_metrics_inputs(
    labels: list[float] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(labels_np) & np.isfinite(scores_np)
    labels_np = labels_np[finite]
    scores_np = np.clip(scores_np[finite], 0.0, 1.0)
    labels_np = (labels_np >= 0.5).astype(np.int64)
    return labels_np, scores_np


def compute_metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {"loss": float("nan"), "accuracy": float("nan"), "auroc": float("nan")}
    labels_np, scores_np = sanitize_binary_metrics_inputs(labels, scores)
    if labels_np.size == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "auroc": float("nan")}
    preds_np = (scores_np >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "mean_positive_score": float(scores_np[labels_np == 1].mean()) if (labels_np == 1).any() else float("nan"),
        "mean_negative_score": float(scores_np[labels_np == 0].mean()) if (labels_np == 0).any() else float("nan"),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels_np, scores_np))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def maybe_cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def resolve_eval_device(config: dict[str, Any], train_device: torch.device) -> torch.device:
    requested = str(config["eval"].get("device", train_device.type)).lower()
    if requested == train_device.type:
        return train_device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("eval.device=cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("eval.device=mps requested but MPS is unavailable.")
        return torch.device("mps")
    raise RuntimeError(f"Unsupported eval.device override: {requested}")


def evaluate(
    model: PairwiseDifferenceModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    labels: list[float] = []
    scores: list[float] = []
    skipped_nonfinite = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            debug_log({"phase": "evaluate_batch_start", "batch_index": batch_index, "device": device.type})
            batch = move_batch(batch, device, include_label=False)
            output = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
            if output.logits.numel() == 0:
                debug_log({"phase": "evaluate_batch_empty", "batch_index": batch_index})
                continue
            valid_indices_cpu = output.valid_indices.detach().cpu()
            labels_batch_cpu = batch["label"].index_select(0, valid_indices_cpu)
            labels_batch = labels_batch_cpu.to(device)
            finite = torch.isfinite(output.logits) & torch.isfinite(output.prob) & torch.isfinite(labels_batch)
            if not finite.any():
                skipped_nonfinite += int(output.logits.numel())
                debug_log({"phase": "evaluate_batch_nonfinite_only", "batch_index": batch_index})
                continue
            logits_batch = output.logits[finite]
            probs_batch = output.prob[finite]
            labels_batch = labels_batch[finite]
            finite_cpu = finite.detach().cpu()
            labels_batch_cpu = labels_batch_cpu[finite_cpu]
            skipped_nonfinite += int(output.logits.numel() - logits_batch.numel())
            loss = criterion(logits_batch, labels_batch)
            losses.append(loss.item())
            labels.extend(labels_batch_cpu.tolist())
            scores.extend(probs_batch.detach().cpu().tolist())
            debug_log(
                {
                    "phase": "evaluate_batch_done",
                    "batch_index": batch_index,
                    "count": int(logits_batch.numel()),
                    "loss": float(loss.item()),
                }
            )
    metrics = compute_metrics(labels, scores)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["skipped_nonfinite"] = float(skipped_nonfinite)
    return metrics


def save_checkpoint(
    path: Path,
    model: PairwiseDifferenceModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def append_epoch_metrics(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = output_dir / "epoch_metrics.jsonl"

    device = make_device()
    eval_device = resolve_eval_device(config, device)
    model = build_model(config).to(device)
    eval_model = build_model(config).to(eval_device) if eval_device != device else None
    train_dataset = build_dataset(config, split="train")
    val_split = config["data"].get("val_split", "val")
    val_dataset = build_dataset(config, split=val_split)
    criterion = build_criterion(config, device, train_dataset=train_dataset)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    start_epoch = 0
    best_metric = float("-inf")
    resume_path = Path(args.resume) if args.resume else None
    if resume_path:
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])

    use_amp = bool(config["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    primary_metric = config["eval"].get("primary_metric", "auroc")
    log_every_steps = int(config["train"].get("log_every_steps", 100))
    checkpoint_every_steps = int(config["train"].get("checkpoint_every_steps", 0))
    max_grad_norm = config["train"].get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    total_epochs = int(config["train"]["epochs"])

    train_epoch_subsets = prepare_epoch_subsets(config, train_dataset, "train", total_epochs)
    val_epoch_subsets = prepare_epoch_subsets(config, val_dataset, val_split, 1)

    for epoch in range(start_epoch, total_epochs):
        train_loader = build_loader(
            config,
            dataset=train_dataset,
            split="train",
            shuffle=True,
            epoch_index=epoch,
            epoch_subsets=train_epoch_subsets,
        )
        val_loader = build_loader(
            config,
            dataset=val_dataset,
            split=val_split,
            shuffle=False,
            epoch_index=0,
            epoch_subsets=val_epoch_subsets,
        )
        model.train()
        train_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device, include_label=False)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast_ctx:
                output = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
                if output.logits.numel() == 0:
                    continue
                valid_indices_cpu = output.valid_indices.detach().cpu()
                labels_batch_cpu = batch["label"].index_select(0, valid_indices_cpu)
                labels_batch = labels_batch_cpu.to(device)
                logits_batch, probs_batch, labels_batch = filter_finite_batch_tensors(
                    output.logits,
                    output.prob,
                    labels_batch,
                )
                if logits_batch.numel() == 0:
                    continue
                loss = criterion(logits_batch, labels_batch)
            if not torch.isfinite(loss):
                print(json.dumps({"epoch": epoch, "step": step + 1, "warning": "skipping_nonfinite_loss"}), flush=True)
                continue
            if use_amp:
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            json.dumps({"epoch": epoch, "step": step + 1, "warning": "skipping_nonfinite_grad_norm"}),
                            flush=True,
                        )
                        continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            json.dumps({"epoch": epoch, "step": step + 1, "warning": "skipping_nonfinite_grad_norm"}),
                            flush=True,
                        )
                        continue
                optimizer.step()
            train_losses.append(loss.item())
            global_step = epoch * max(len(train_loader), 1) + step
            writer.add_scalar("train/loss", loss.item(), global_step)
            if log_every_steps > 0 and ((step + 1) % log_every_steps == 0 or step == 0):
                print(json.dumps({"epoch": epoch, "step": step + 1, "train_loss": loss.item()}), flush=True)
            if checkpoint_every_steps > 0 and (step + 1) % checkpoint_every_steps == 0:
                save_checkpoint(output_dir / "latest_step.pt", model, optimizer, epoch, best_metric, config)

        debug_log({"phase": "train_epoch_done", "epoch": epoch})
        maybe_cleanup_device(device)
        debug_log({"phase": "after_train_cleanup", "epoch": epoch, "device": device.type})
        if eval_model is not None:
            eval_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            eval_model.load_state_dict(eval_state)
            eval_criterion = build_criterion(config, eval_device)
            val_metrics = evaluate(eval_model, val_loader, eval_criterion, eval_device)
        else:
            debug_log({"phase": "before_eval", "epoch": epoch, "device": device.type})
            val_metrics = evaluate(model, val_loader, criterion, device)
            debug_log({"phase": "after_eval", "epoch": epoch, "device": device.type, "val_metrics": val_metrics})
        maybe_cleanup_device(device)
        debug_log({"phase": "after_eval_cleanup", "epoch": epoch, "device": device.type})
        if eval_device != device:
            maybe_cleanup_device(eval_device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        metric_value = val_metrics.get(primary_metric, float("nan"))
        if not np.isnan(metric_value) and metric_value >= best_metric:
            best_metric = metric_value
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_metric, config)

        latest_path = output_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, best_metric, config)
        summary = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        append_epoch_metrics(metrics_log_path, summary)
        print(json.dumps(summary))

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="GOBC/configs/base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
