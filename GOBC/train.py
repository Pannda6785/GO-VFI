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
    return VimeoOverlayPairDataset(
        dataset_root=config["data"]["dataset_root"],
        split=split,
        image_size=config["data"]["image_size"],
        crop_margin=config["data"]["crop_margin"],
        normalize=True,
        patch_size=patch_size,
        min_patch_tokens=min_patch_tokens,
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
    if len(indices) < count:
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
        valid_indices = _collect_valid_indices(dataset, required, subset_seed, progress_label=split)
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
    )


def build_criterion(config: dict[str, Any], device: torch.device) -> nn.Module:
    pos_weight_value = float(config["train"].get("pos_weight", 1.0))
    if pos_weight_value == 1.0:
        return nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("image1", "image2", "mask1", "mask2", "label"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def compute_metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {"loss": float("nan"), "accuracy": float("nan"), "auroc": float("nan")}
    labels_np = np.asarray(labels, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    preds_np = (scores_np >= 0.5).astype(np.float32)
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
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
            if output.logits.numel() == 0:
                continue
            labels_batch = batch["label"].index_select(0, output.valid_indices)
            loss = criterion(output.logits, labels_batch)
            losses.append(loss.item())
            labels.extend(labels_batch.detach().cpu().tolist())
            scores.extend(output.prob.detach().cpu().tolist())
    metrics = compute_metrics(labels, scores)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
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
    criterion = build_criterion(config, device)
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
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with autocast_ctx:
                output = model(batch["image1"], batch["mask1"], batch["image2"], batch["mask2"])
                if output.logits.numel() == 0:
                    continue
                labels_batch = batch["label"].index_select(0, output.valid_indices)
                loss = criterion(output.logits, labels_batch)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            global_step = epoch * max(len(train_loader), 1) + step
            writer.add_scalar("train/loss", loss.item(), global_step)
            if log_every_steps > 0 and ((step + 1) % log_every_steps == 0 or step == 0):
                print(json.dumps({"epoch": epoch, "step": step + 1, "train_loss": loss.item()}), flush=True)
            if checkpoint_every_steps > 0 and (step + 1) % checkpoint_every_steps == 0:
                save_checkpoint(output_dir / "latest_step.pt", model, optimizer, epoch, best_metric, config)

        maybe_cleanup_device(device)
        if eval_model is not None:
            eval_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            eval_model.load_state_dict(eval_state)
            eval_criterion = build_criterion(config, eval_device)
            val_metrics = evaluate(eval_model, val_loader, eval_criterion, eval_device)
        else:
            val_metrics = evaluate(model, val_loader, criterion, device)
        maybe_cleanup_device(device)
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
