# GORe

`GORe` is a local overlay-refinement module, not a full video frame interpolation system.

Its job is to take:

- `I0`
- `I0.5_cp`
- overlay masks `M_s`

and produce a refined midpoint frame that only edits the overlay neighborhood.

## Design Contract

The implementation in this folder is grounded on the following design:

Input:

- `I0`: source/context frame at time `0`
- `I0.5_cp`: copy-pasted midpoint candidate
- `M`: union overlay mask
- `E`: editable zone
- `T`: aggregated transition band

Output:

- `delta`: predicted RGB correction
- `P = I0.5_cp + delta`: refined RGB content
- final frame:
  - `I_hat_0.5 = (1 - E) * I0.5_cp + E * P`

Training target:

- `I0.5_gt`

Loss:

- `L_E = masked_l1(I_hat_0.5, I0.5_gt, E)`
- `L_T = masked_l1(I_hat_0.5, I0.5_gt, T)`
- `L_grad = gradient_loss(I_hat_0.5, I0.5_gt, T)`
- total:
  - `L = L_E + lambda_T * L_T + beta * L_grad`

`I0` is used only as conditioning/context. This module does not explicitly copy from `I0`.

## File Layout

- `core.py`: model, loss, and runtime helpers
- `data.py`: dataset loader for the refinement-training dataset
- `run.py`: one runner with `train` and `infer` subcommands

This is intentionally smaller than the previous split package layout.

## Model

The code now exposes two clean model presets:

- `small`: current baseline
- `8m`: larger 4-scale proposal

Both follow exactly the same method contract. Only the U-Net capacity changes.

Input channels:

- `I0`: 3
- `I0.5_cp`: 3
- `M`: 1
- `E`: 1
- `T`: 1

Total:

- `9` channels

The network predicts a residual RGB correction and refines `I0.5_cp` as `P = I0.5_cp + delta`. The final output is then composed with `E`.

Model presets:

- `small`
  - stages: `32, 64, 128`
  - bottleneck: `256`
- `8m`
  - stages: `32, 64, 128, 256`
  - bottleneck: `448`

## Dataset Assumption

Current dataset root:

- `Datasets/vimeo_aug_interpolated_for_refinement_training`

Per sample, the loader expects:

- `I0.png`
- `I_0.5_inter_copied.png`
- `I_0.5_copied.png`
- `M.png`
- `peroverlay/*.png`
- split-level `_manifest.json`

Current mapping used by this implementation:

- `I0 = I0.png`
- `I0.5_cp = I_0.5_inter_copied.png`
- `I0.5_gt = I_0.5_copied.png`
- `M = M.png`

The dataset manifest already exposes selected overlay ids per sample. Those ids are used to build `T` from `peroverlay/<object_id>_I0.png` and `peroverlay/<object_id>_I1.png`.

## Mask Construction

Given the exposed masks, the loader uses:

Union mask:

- `M = M.png`

Editable zone:

- `E = dilate(M, r_e)`

Aggregated transition band:

- `T = union_s (dilate(M_s, r2) - erode(M_s, r1))`
- each `M_s` is built as `union(mask_s at I0, mask_s at I1)` for manifest-selected overlays

The current defaults are:

- `r_e = 8`
- `r1 = 1`
- `r2 = 6`

These are CLI-configurable.

## How To Train

From the repo root:

```bash
.venv/bin/python -m GORe.run train \
  --dataset-root Datasets/vimeo_aug_interpolated_for_refinement_training \
  --model small \
  --epochs 1 \
  --batch-size 2 \
  --limit 16
```

Useful options:

- `--editable-radius`
- `--transition-erode-radius`
- `--transition-dilate-radius`
- `--model`
- `--lambda-t`
- `--beta-grad`

Checkpoints are saved to:

- `GORe/checkpoints/`

## How To Infer

```bash
.venv/bin/python -m GORe.run infer \
  --dataset-root Datasets/vimeo_aug_interpolated_for_refinement_training \
  --model 8m \
  --checkpoint GORe/checkpoints/gore_epoch_001.pt \
  --sample-index 0
```

Outputs are written by default to:

- `Test/GORe/`

Saved outputs include:

- `i05_cp`
- `i05_gt`
- predicted patch
- final prediction
- `M`
- `E`
- `T`

## Current Scope

This folder currently implements:

- the local refinement network
- direct loading of the refinement-training dataset
- training
- inference

It does not yet implement:

- final evaluation protocol
- experiment logging beyond stdout and checkpoints
- distributed training

If evaluation details change later, update the training or evaluation scripts without changing the basic design contract above.
