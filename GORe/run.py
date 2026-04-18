from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from .core import OverlayRefinerLoss, autocast_context, build_model, make_runtime
from .data import MaskConfig, VimeoOverlayRefineDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GORe runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", type=Path, default=Path("Datasets/vimeo_aug_interpolated_for_refinement_training"))
    common.add_argument("--split", default="train")
    common.add_argument("--model", choices=("small", "8m"), default="small")
    common.add_argument("--editable-radius", type=int, default=8)
    common.add_argument("--transition-erode-radius", type=int, default=1)
    common.add_argument("--transition-dilate-radius", type=int, default=6)

    train = subparsers.add_parser("train", parents=[common], help="Train GORe")
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--lambda-e", type=float, default=1.0)
    train.add_argument("--beta-grad", type=float, default=0.5)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--save-dir", type=Path, default=Path("GORe/checkpoints"))
    train.add_argument("--train-size", type=int, default=270)
    train.add_argument("--val-size", type=int, default=30)
    train.add_argument("--val-batch-size", type=int, default=4)
    train.add_argument("--val-output-dir", type=Path, default=Path("GORe/val_outputs"))
    train.add_argument("--plot-path", type=Path, default=Path("GORe/loss_plot.png"))
    train.add_argument("--metrics-path", type=Path, default=Path("GORe/metrics.json"))
    train.add_argument("--seed", type=int, default=0)

    infer = subparsers.add_parser("infer", parents=[common], help="Run GORe inference")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--sample-index", type=int, default=0)
    infer.add_argument("--output-dir", type=Path, default=Path("Test/GORe"))

    return parser.parse_args()


def make_mask_config(args: argparse.Namespace) -> MaskConfig:
    return MaskConfig(
        editable_radius=args.editable_radius,
        transition_erode_radius=args.transition_erode_radius,
        transition_dilate_radius=args.transition_dilate_radius,
    )


def serializable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def to_mask_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")


def split_dataset(dataset: VimeoOverlayRefineDataset, train_size: int, val_size: int) -> tuple[Subset, Subset]:
    total = len(dataset)
    if train_size + val_size > total:
        raise ValueError(f"Requested split {train_size}+{val_size} exceeds dataset size {total}")
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: OverlayRefinerLoss, loader: DataLoader, runtime) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_global": 0.0, "loss_e": 0.0, "loss_grad": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, runtime.device)
        with autocast_context(runtime):
            outputs = model(
                i0=batch["i0"],
                i05_cp=batch["i05_cp"],
                m=batch["m"],
                e=batch["e"],
                t=batch["t"],
            )
            losses = criterion(
                pred=outputs["pred"],
                target=batch["i05_gt"],
                e=batch["e"],
                t=batch["t"],
            )
        count += 1
        for key in totals:
            totals[key] += float(losses[key].item())
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def dump_validation_predictions(model: torch.nn.Module, loader: DataLoader, runtime, output_dir: Path, epoch: int) -> None:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_counter = 0
    for batch in loader:
        batch_on_device = move_batch(batch, runtime.device)
        with autocast_context(runtime):
            outputs = model(
                i0=batch_on_device["i0"],
                i05_cp=batch_on_device["i05_cp"],
                m=batch_on_device["m"],
                e=batch_on_device["e"],
                t=batch_on_device["t"],
            )
        batch_size = batch["i0"].shape[0]
        for i in range(batch_size):
            stem = str(batch["sample_id"][i]).replace("/", "__")
            to_rgb_image(batch["i05_cp"][i]).save(epoch_dir / f"{stem}_i05_cp.png")
            to_rgb_image(batch["i05_gt"][i]).save(epoch_dir / f"{stem}_i05_gt.png")
            to_rgb_image(outputs["pred_patch"][i]).save(epoch_dir / f"{stem}_pred_patch.png")
            to_rgb_image(outputs["pred"][i]).save(epoch_dir / f"{stem}_pred.png")
            to_mask_image(batch["m"][i]).save(epoch_dir / f"{stem}_M.png")
            to_mask_image(batch["e"][i]).save(epoch_dir / f"{stem}_E.png")
            to_mask_image(batch["t"][i]).save(epoch_dir / f"{stem}_T.png")
            sample_counter += 1
    print(f"saved_val_outputs={epoch_dir} count={sample_counter}")


def save_loss_plot(history: dict[str, list[float]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(len(history["val_loss"])))
    plt.figure(figsize=(8, 5))
    if history["train_loss"]:
        plt.plot(epochs[1:], history["train_loss"], label="train_loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="val_loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GORe Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def run_train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    runtime = make_runtime()
    dataset = VimeoOverlayRefineDataset(
        root=args.dataset_root,
        split=args.split,
        mask_config=make_mask_config(args),
        limit=args.limit,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.train_size, args.val_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(args.model).to(runtime.device)
    criterion = OverlayRefinerLoss(lambda_e=args.lambda_e, beta_grad=args.beta_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.val_output_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    print(
        json.dumps(
            {
                "device": str(runtime.device),
                "model": args.model,
                "params": parameter_count(model),
                "samples": len(dataset),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "batch_size": args.batch_size,
                "mask_config": make_mask_config(args).__dict__,
            },
            indent=2,
        )
    )

    val_metrics = evaluate(model, criterion, val_loader, runtime)
    history["val_loss"].append(val_metrics["loss"])
    dump_validation_predictions(model, val_loader, runtime, args.val_output_dir, 0)
    print(
        f"epoch_summary epoch=0 "
        f"val_loss={val_metrics['loss']:.6f} "
        f"val_loss_global={val_metrics['loss_global']:.6f} "
        f"val_loss_e={val_metrics['loss_e']:.6f} "
        f"val_loss_grad={val_metrics['loss_grad']:.6f}"
    )

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_steps = 0
        for batch in train_loader:
            step += 1
            epoch_train_steps += 1
            batch = move_batch(batch, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                outputs = model(
                    i0=batch["i0"],
                    i05_cp=batch["i05_cp"],
                    m=batch["m"],
                    e=batch["e"],
                    t=batch["t"],
                )
                losses = criterion(
                    pred=outputs["pred"],
                    target=batch["i05_gt"],
                    e=batch["e"],
                    t=batch["t"],
                )
            losses["loss"].backward()
            optimizer.step()
            epoch_train_loss += float(losses["loss"].item())
            print(
                f"epoch={epoch + 1} step={step} "
                f"loss={float(losses['loss'].item()):.6f} "
                f"loss_global={float(losses['loss_global'].item()):.6f} "
                f"loss_e={float(losses['loss_e'].item()):.6f} "
                f"loss_grad={float(losses['loss_grad'].item()):.6f}"
            )

        train_loss = epoch_train_loss / max(epoch_train_steps, 1)
        val_metrics = evaluate(model, criterion, val_loader, runtime)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        dump_validation_predictions(model, val_loader, runtime, args.val_output_dir, epoch + 1)

        checkpoint_path = args.save_dir / f"gore_epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": serializable_args(args),
                "mask_config": make_mask_config(args).__dict__,
                "history": history,
            },
            checkpoint_path,
        )
        print(
            f"epoch_summary epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_loss_global={val_metrics['loss_global']:.6f} "
            f"val_loss_e={val_metrics['loss_e']:.6f} "
            f"val_loss_grad={val_metrics['loss_grad']:.6f}"
        )
        print(f"saved_checkpoint={checkpoint_path}")

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(history, indent=2))
    save_loss_plot(history, args.plot_path)
    print(f"saved_metrics={args.metrics_path}")
    print(f"saved_plot={args.plot_path}")


@torch.no_grad()
def run_infer(args: argparse.Namespace) -> None:
    runtime = make_runtime()
    dataset = VimeoOverlayRefineDataset(
        root=args.dataset_root,
        split=args.split,
        mask_config=make_mask_config(args),
    )
    sample = dataset[args.sample_index]
    model = build_model(args.model).to(runtime.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = model(
        i0=sample["i0"].unsqueeze(0).to(runtime.device),
        i05_cp=sample["i05_cp"].unsqueeze(0).to(runtime.device),
        m=sample["m"].unsqueeze(0).to(runtime.device),
        e=sample["e"].unsqueeze(0).to(runtime.device),
        t=sample["t"].unsqueeze(0).to(runtime.device),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(sample["sample_id"]).replace("/", "__")
    to_rgb_image(sample["i05_cp"]).save(args.output_dir / f"{stem}_i05_cp.png")
    to_rgb_image(sample["i05_gt"]).save(args.output_dir / f"{stem}_i05_gt.png")
    to_rgb_image(outputs["pred_patch"][0]).save(args.output_dir / f"{stem}_pred_patch.png")
    to_rgb_image(outputs["pred"][0]).save(args.output_dir / f"{stem}_pred.png")
    to_mask_image(sample["m"]).save(args.output_dir / f"{stem}_M.png")
    to_mask_image(sample["e"]).save(args.output_dir / f"{stem}_E.png")
    to_mask_image(sample["t"]).save(args.output_dir / f"{stem}_T.png")
    print(f"saved_outputs={args.output_dir}")
    print(f"sample_id={sample['sample_id']}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
