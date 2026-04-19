# GOBC

GOBC is a minimal pairwise structural-difference baseline for masked overlay objects in `Datasets/vimeo_triplet_augmented_full`.

## Data contract

The dataset is consumed **per overlay**, not per composite frame.

Each sample uses:

- `image1 = I0.png`
- `image2 = I1.png`
- `mask1 = overlays_masks/{object_id}_I0.png`
- `mask2 = overlays_masks/{object_id}_I1.png`

The dataset loader applies a shared union crop from `mask1` and `mask2`, expands it by a fixed context margin, resizes both images and masks to a fixed square size, and normalizes images with DINO-compatible ImageNet statistics.

### Filtering and labels

- Drop any sequence whose `metadata.json` reports `scenechange`.
- Drop any overlay whose `temporal.mode == "appear_disappear"`.
- Label `1 = different` iff:
  - `temporal.mode == "change_appearance"`
  - `temporal.detail.variant == "textual"`
- Label `0 = similar` otherwise.

This intentionally does **not** reuse the original discontinuity label blindly.

## Model

- frozen DINOv2 backbone
- shared linear projection to a smaller working dimension
- one cross-attention block in each direction
- masked mean pooling over object tokens only
- relation head over `[z1, z2, |z1-z2|, z1*z2]`

The forward API is:

```python
out = model(image1, mask1, image2, mask2)
```

`out` contains:

- `logits: [B]`
- `prob: [B]`
- projected patch tokens and patch masks for debugging

## Usage

Train:

```bash
.venv/bin/python GOBC/train.py --config GOBC/configs/base.yaml
```

Evaluate:

```bash
.venv/bin/python GOBC/eval.py \
  --config GOBC/configs/base.yaml \
  --checkpoint GOBC/outputs/base/best.pt \
  --split val
```

Run tests:

```bash
.venv/bin/python -m unittest GOBC.tests.test_shapes
```
