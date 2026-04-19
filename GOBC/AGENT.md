## Objective

Implement a **pairwise structural difference classifier** for masked objects:

* input: `(img1, mask1, img2, mask2)`
* backbone: **frozen DINOv2**
* comparator: **minimal cross-attention over patch tokens**
* output: probability that the two objects are **structurally different**

This is grounded in current DINOv2 interfaces: the official repo exposes PyTorch Hub endpoints like `dinov2_vitb14`, and Hugging Face exposes `Dinov2Model.from_pretrained(...)`. DINOv2 is intended as a general-purpose visual feature backbone. ([GitHub][1])

## Non-goals

Do **not**:

* build a full training framework with ablations on day 1
* add HED, graph matching, deformable attention, or localization heads
* fine-tune DINOv2 initially
* invent synthetic data generation until the baseline is trainable and inspectable

## Deliverables

The agent should produce these files:

* `models/dinov2_backbone.py`
* `models/cross_attention_matcher.py`
* `data/pair_dataset.py`
* `train.py`
* `eval.py`
* `configs/base.yaml`
* `tests/test_shapes.py`
* `README.md`

## Implementation phases

### Phase 1: Backbone wrapper

Implement a wrapper with **one stable interface** regardless of source:

```python
tokens, grid_hw = backbone(images)
```

Requirements:

* support `source=torchhub` and `source=hf`
* default to `dinov2_vitb14`
* return **patch tokens only**, not CLS
* infer patch grid `(Gh, Gw)` from token count
* freeze backbone by default

Use official loading paths only:

* `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
* `Dinov2Model.from_pretrained("facebook/dinov2-base")` ([GitHub][1])

Acceptance criteria:

* for input `[B, 3, H, W]`, output is `[B, N, C]`
* `N == Gh * Gw`
* both torchhub and HF codepaths pass the same shape test

### Phase 2: Object-mask tokenization

Implement utilities to map masks to patch tokens:

* resize mask to patch grid using area interpolation
* flatten to `[B, N]`
* keep both:
  * a soft patch mask from the projected area values
  * a hard patch mask from `soft_mask > threshold`

Do **not** do boundary/interior splitting.
Only use the mask as:

* object vs non-object selector
* optional crop generator

Acceptance criteria:

* mask projection aligns with patch grid
* empty-mask case raises a clear error or is skipped in dataset

### Phase 3: Minimal cross-attention comparator

Implement the comparator with this structure:

1. DINO patch tokens for object 1 and object 2
2. linear projection from DINO dim to working dim, e.g. `256`
3. one cross-attention block `1 -> 2`
4. one cross-attention block `2 -> 1`
5. masked mean summary over object tokens
6. relation head over `[z1, z2, |z1-z2|, z1*z2]`
7. scalar logit

Keep it minimal:

* one cross-attention block per direction
* one FFN inside each block
* one small MLP head

Do **not** flatten all patch tokens into one huge vector.

Acceptance criteria:

* forward pass works on dummy batch
* outputs:

  * `logits: [B]`
  * `prob: [B]`
  * optional debug tensors `tokens1`, `tokens2`, `mask1`, `mask2`

### Phase 4: Data pipeline

Implement a dataset that yields:

```python
{
  "image1": Tensor[3,H,W],
  "mask1":  Tensor[1,H,W],
  "image2": Tensor[3,H,W],
  "mask2":  Tensor[1,H,W],
  "label":  Tensor[]   # 1=different, 0=same  OR choose one convention and keep it fixed
}
```

Rules:

* crop from the union box of `mask1` and `mask2`, then apply a fixed context margin
* resize to backbone input size
* normalize images using the chosen DINO preprocessing path
* keep one label convention throughout the repo and write it in README

Keep the first version simple:

* no online hard-negative mining yet
* no synthetic edits yet unless your dataset already needs them

Acceptance criteria:

* a batch loads and trains
* visual debug script can save sampled pairs with masks and labels

## Data specification

Use `Datasets/vimeo_triplet_augmented_full` as a **per-overlay** dataset, not a per-composite dataset.

Each training example corresponds to one overlay object from one sequence entry:

* `image1 = I0.png`
* `image2 = I1.png`
* `mask1 = overlays_masks/{object_id}_I0.png`
* `mask2 = overlays_masks/{object_id}_I1.png`
* label derived from the overlay's temporal metadata

Read labels from `metadata.json` using the `overlays` entries, not from a class-name heuristic.

Sequence-level filtering:

* drop the entire sequence if `scenechange` is non-null or otherwise indicates a scene change
* skip entries with no overlays or no valid classification samples

Overlay-level filtering and labeling:

* drop overlays where `temporal.mode == "appear_disappear"`
* drop overlays where the endpoint center-of-mass metadata is too close to the frame edge:
  * `center0` must lie at least `frame_width / 16` away from every edge of `I0`
  * `center1` must lie at least `frame_width / 16` away from every edge of `I1`
* label `different` iff:
  * `temporal.mode == "change_appearance"`
  * and either:
    * `temporal.detail.variant == "textual"`
    * or `temporal.detail.variant == "composite"` and `temporal.detail.components` contains `"textual"`
* otherwise label `similar`

Label convention:

* `1 = different`
* `0 = similar`

Implementation notes:

* prefer `overlays[*].temporal` as the source of truth for mode and variant
* do not reuse the dataset's original discontinuity label blindly if it conflicts with the rule above
* do not infer textual change from `goons.class_name`; use `temporal.detail.variant` and, for composite overlays, `temporal.detail.components`
* keep the pairing endpoint-only for GOBC v1: `(I0, mask_I0)` vs `(I1, mask_I1)`
* for train sampling only, allow a non-duplicating random endpoint swap:
  * with probability `p`, feed `(I1, mask_I1, I0, mask_I0)` instead of `(I0, mask_I0, I1, mask_I1)`
  * do not duplicate the dataset index for this; apply it at sample fetch time
* use a shared union crop for both frames so the pair stays spatially aligned
* design the dataset and collate path for batched training from the start; do not build a single-sample-only pipeline first

### Phase 5: Training loop

Implement a plain training script:

* BCE-with-logits on pair label
* AdamW
* frozen backbone
* train only projection, cross-attention blocks, and head
* validation every epoch
* save best checkpoint by val AUROC or val accuracy

Metrics to log:

* loss
* accuracy
* AUROC
* mean positive score
* mean negative score

Why AUROC:

* threshold-independent
* better signal early in pairwise classification

Acceptance criteria:

* one config runs end-to-end
* logs and checkpoints are produced
* resume from checkpoint works

### Phase 6: Evaluation and debugging

Implement `eval.py` with:

* checkpoint loading
* validation/test metrics
* confusion matrix
* score histogram for same vs different
* optional nearest-neighbor sanity check from saved embeddings

Also implement a debug mode that saves:

* patch-mask coverage stats
* token count per object
* examples with the highest-confidence mistakes

Acceptance criteria:

* can inspect whether failures come from data, masks, or comparator

## Design rules the agent must follow

### 1. Keep one clean model API

Use:

```python
out = model(image1, mask1, image2, mask2)
```

No multiple forward modes unless strictly necessary.

### 2. Separate backbone from comparator

The backbone wrapper should be swappable without touching the comparator.

### 3. Freeze backbone first

Start with frozen DINOv2 because the model is meant to provide reusable visual features, and many downstream setups benefit from training only lightweight heads first. ([GitHub][1])

### 4. Prefer inspectability over cleverness

Expose intermediate tensors and keep the architecture small enough that failures are interpretable.

### 5. No “temporary” shortcuts that become permanent

Avoid:

* magic constants scattered through code
* hidden label flips
* duplicated preprocessing logic
* mixed conventions between torchhub and HF paths

## Suggested config for v1

```yaml
model:
  backbone_source: torchhub
  backbone_name: dinov2_vitb14
  freeze_backbone: true
  proj_dim: 256
  num_heads: 8
  dropout: 0.0

data:
  image_size: 518
  crop_margin: 0.15
  mask_threshold: 0.3

train:
  batch_size: 16
  epochs: 20
  lr: 1e-4
  weight_decay: 1e-4
  num_workers: 4
  amp: true

eval:
  primary_metric: auroc
```

## Required tests

The agent should add tests for:

* backbone output shape on both loading paths
* patch-grid inference correctness
* mask resizing and flattening correctness
* cross-attention matcher forward shape
* training step backward pass
* checkpoint save/load consistency

## Milestones

### Milestone 1

Backbone wrapper + shape tests only.

### Milestone 2

Cross-attention matcher runs on dummy tensors.

### Milestone 3

Dataset yields real batches and training starts.

### Milestone 4

One full epoch on real data with metrics and checkpointing.

### Milestone 5

Evaluation script and debug artifacts.

## What to tell the agent explicitly

Use this as the task brief:

> Implement a minimal but production-minded pairwise structural difference classifier for masked objects. Use frozen DINOv2 patch tokens from either official PyTorch Hub or Hugging Face, project tokens to a smaller dimension, apply one cross-attention block in each direction between the two objects, summarize only masked object tokens, and classify with a small relation head. Keep modules separated, typed, tested, and easy to inspect. Do not add extra research ideas, ablations, or synthetic data machinery in this pass.

## First commit sequence

1. `feat: add DINOv2 backbone wrapper with torchhub/hf support`
2. `feat: add patch-mask projection utilities`
3. `feat: add minimal cross-attention matcher`
4. `feat: add pair dataset and preprocessing`
5. `feat: add training and evaluation scripts`
6. `test: add shape and checkpoint tests`
7. `docs: add usage and design notes`

## One important implementation note

Prefer the **official torchhub path first** unless your environment is already standardized on Transformers. The official repo is the clearest reference for the `dinov2_vit*14` endpoints, while Hugging Face is convenient if you already rely on their model stack. ([GitHub][1])

[1]: https://github.com/facebookresearch/dinov2?utm_source=chatgpt.com "PyTorch code and models for the DINOv2 self-supervised ..."
