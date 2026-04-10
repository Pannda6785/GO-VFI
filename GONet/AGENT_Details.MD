For Codex to implement this well, you want a **much more detailed PyTorch spec** with:

* block responsibilities
* tensor contracts
* per-module inputs/outputs
* what is learned vs deterministic
* object-level vs frame-level data flow

So below is a **detailed architecture design** you can hand to Codex.

---

# Detailed PyTorch architecture

We will design this around **per-object processing** plus **frame-level composition**.

The full model is:

[
\texttt{GOHandlerNet} =
\texttt{OverlayCropBuilder}
\rightarrow
\texttt{InstanceBatchBuilder}
\rightarrow
\texttt{TemporalInterpreter}
\rightarrow
\texttt{MidpointWarpRenderer}
\rightarrow
\texttt{CanvasComposer}
\rightarrow
\texttt{BoundaryRefiner}
]

---

# 1. Data contract

The most important thing is to define the interfaces clearly.

## Frame-level inputs

```python
I0:     FloatTensor[B, 3, H, W]
I1:     FloatTensor[B, 3, H, W]
I0_bg:  FloatTensor[B, 3, H, W]
I1_bg:  FloatTensor[B, 3, H, W]
I_bg:   FloatTensor[B, 3, H, W]
```

## Correspondence structure

For each batch element, `C[b]` is a list of GO instances.
Each GO instance contains:

```python
{
    "mask0": FloatTensor[H, W] or None,
    "mask1": FloatTensor[H, W] or None,
    "id": int,
    # optional metadata
}
```

For implementation, Codex should normalize this into a packed instance representation.

---

# 2. Packed per-instance representation

Because the network reasons per overlay, it should convert all instances across the batch into a packed tensor batch.

## Packed object tensors

For `N` total GO instances across the minibatch:

```python
crop_I0:    FloatTensor[N, 3, Hp, Wp]
crop_I1:    FloatTensor[N, 3, Hp, Wp]
crop_I0_bg: FloatTensor[N, 3, Hp, Wp]
crop_I1_bg: FloatTensor[N, 3, Hp, Wp]
crop_M0:    FloatTensor[N, 1, Hp, Wp]
crop_M1:    FloatTensor[N, 1, Hp, Wp]
```

Recommended crop size:

```python
Hp = Wp = 192
```

Also keep placement metadata:

```python
instance_batch_idx: LongTensor[N]        # which frame batch item this instance belongs to
instance_boxes_xyxy: FloatTensor[N, 4]   # crop region in original image coordinates
instance_valid0: BoolTensor[N]
instance_valid1: BoolTensor[N]
```

---

# 3. Top-level module layout

Here is the exact module tree I recommend:

```python
class GOHandlerNet(nn.Module):
    def __init__(self, cfg):
        self.crop_builder
        self.instance_encoder
        self.temporal_interpreter
        self.midpoint_renderer
        self.canvas_composer
        self.boundary_refiner
```

Now I’ll define each in detail.

---

# 4. OverlayCropBuilder

This is deterministic preprocessing.

## Responsibility

* convert frame-level masks/correspondences into packed object crops
* extract overlay residuals
* build geometric metadata

## Input

```python
I0, I1, I0_bg, I1_bg, C
```

## Output

A dictionary:

```python
{
    "crop_I0": ...,
    "crop_I1": ...,
    "crop_I0_bg": ...,
    "crop_I1_bg": ...,
    "crop_M0": ...,
    "crop_M1": ...,
    "crop_G0": ...,
    "crop_G1": ...,
    "meta": ...,
    "placement": ...
}
```

where

[
G_0 = (I_0 - I_{0,bg}) \odot M_0,\qquad
G_1 = (I_1 - I_{1,bg}) \odot M_1
]

## Metadata vector

For each instance:

```python
meta: FloatTensor[N, Dm]
```

Recommended contents:

* valid0 flag
* valid1 flag
* centroid0 x,y
* centroid1 x,y
* delta centroid
* area0
* area1
* log(area1 / area0)
* bbox width/height stats
* IoU if both exist

A good default:

```python
Dm = 16
```

---

# 5. Per-instance input tensor design

The interpreter should not only see raw RGB. It should see overlay-centric signals.

For each instance, build:

```python
X0 = concat([
    crop_G0,       # [3]
    crop_M0,       # [1]
    crop_I0,       # [3]
    crop_I0_bg,    # [3]
], dim=channel)

X1 = concat([
    crop_G1,
    crop_M1,
    crop_I1,
    crop_I1_bg,
], dim=channel)
```

So each has:

```python
Cin = 10
```

This is much better than just `G + M`, because:

* `G` captures overlay residual
* `I` captures exact observed edge pixels
* `I_bg` gives compositing context

---

# 6. InstanceEncoder

This should be Siamese and multi-scale.

## Goal

Extract:

* local appearance features
* geometric/mask features
* a global descriptor for mode reasoning

## Recommended architecture

```python
class InstanceEncoder(nn.Module):
    stem
    stage1
    stage2
    stage3
    stage4
    global_pool
```

## Output

For each endpoint:

```python
{
    "f1": FloatTensor[N, C1, 96, 96],
    "f2": FloatTensor[N, C2, 48, 48],
    "f3": FloatTensor[N, C3, 24, 24],
    "f4": FloatTensor[N, C4, 12, 12],
    "g":  FloatTensor[N, Cg],
}
```

Recommended channels:

```python
C1 = 32
C2 = 64
C3 = 128
C4 = 192
Cg = 192
```

## Suggested block type

Use simple Conv-BN-SiLU residual blocks.

A good reusable block:

```python
class ConvNormAct(nn.Module):
    Conv2d -> BatchNorm2d -> SiLU
```

and

```python
class ResidualBlock(nn.Module):
    2x ConvNormAct + skip
```

This is easier for Codex to implement cleanly than relying on timm immediately.

---

# 7. TemporalInterpreter

This is the core learned reasoning module.

It should itself be split into sub-blocks.

```python
class TemporalInterpreter(nn.Module):
    geometry_encoder
    correspondence_block
    fusion_block
    presence_head
    mode_head
    transform_head
    alpha_head
    confidence_head
```

---

## 7.1 GeometryEncoder

This encodes mask/shape/layout cues separately from appearance.

### Input

* `f0["f2"], f1["f2"]`
* `crop_M0, crop_M1`
* `meta`

### Output

A geometry feature vector:

```python
geo_feat: FloatTensor[N, Cgeo]
```

Recommended:

```python
Cgeo = 128
```

This block can be a small CNN over:

* concatenated mid-level features
* masks
* absolute difference
* product features

Example internal tensor:

```python
geo_in = concat([
    f0_f2, f1_f2, abs(f0_f2 - f1_f2), f0_f2 * f1_f2,
    downsample(M0), downsample(M1)
], dim=1)
```

Then:

* conv stack
* global average pool
* concat metadata
* MLP

---

## 7.2 CorrespondenceBlock

This computes cross-frame matching features.

### Input

* global descriptors `g0`, `g1`
* maybe `f3` pooled features

### Output

```python
pair_feat: FloatTensor[N, Cpair]
```

Use standard fusion:

```python
pair = concat([g0, g1, abs(g0-g1), g0*g1], dim=1)
```

Pass through MLP.

Recommended:

```python
Cpair = 256
```

---

## 7.3 FusionBlock

Combines:

* `pair_feat`
* `geo_feat`

and outputs a shared latent:

```python
shared_feat: FloatTensor[N, 256]
```

This is the feature used by the heads.

---

## 7.4 PresenceHead

This predicts whether the GO exists at:

* both frames
* only frame 0
* only frame 1

Output:

```python
presence_logits: FloatTensor[N, 3]
```

Even if you also supervise final mode directly, this head is useful.

---

## 7.5 ModeHead

Final 5-class head:

* static
* appear/disappear
* appearance change
* blink
* affine

Output:

```python
mode_logits: FloatTensor[N, 5]
```

---

## 7.6 TransformHead

This predicts the **midpoint transform**, not endpoint transform.

That is the right target for your pipeline.

### Parameterization

Start with affine 2x3:

```python
transform_params: FloatTensor[N, 6]
```

Interpretation:

```python
[a11, a12, tx, a21, a22, ty]
```

But train as **residual to identity**.

So internally:

```python
raw = head(shared_feat)  # [N, 6]

theta = [
    1 + s * raw[:,0],  s * raw[:,1], raw[:,2],
    s * raw[:,3], 1 + s * raw[:,4], raw[:,5]
]
```

with small scale `s`, like `0.1`.

This stabilizes learning.

---

## 7.7 AlphaHead

For blink / visibility behavior.

Output:

```python
alpha_params: FloatTensor[N, 2]
```

Example interpretation:

* `alpha_scale`
* `alpha_bias`

Or simpler:

```python
alpha_strength: FloatTensor[N, 1]
```

I recommend 2 for flexibility.

---

## 7.8 ConfidenceHead

Optional but useful.

Output:

```python
confidence: FloatTensor[N, 3]
```

For:

* mode confidence
* transform confidence
* render confidence

Not essential for v1, but easy to add.

---

# 8. MidpointWarpRenderer

This is deterministic + mode-conditioned.

```python
class MidpointWarpRenderer(nn.Module):
    def forward(
        self,
        crop_G0, crop_M0,
        transform_params,
        alpha_params,
        mode_logits
    ):
        ...
```

## Responsibilities

* generate midpoint overlay crop
* generate midpoint mask crop
* apply mode-specific logic

## Outputs

```python
{
    "crop_G_mid_base": FloatTensor[N, 3, Hp, Wp],
    "crop_M_mid": FloatTensor[N, 1, Hp, Wp],
}
```

---

## 8.1 Warp operation

Warp both:

* `crop_G0`
* `crop_M0`

using `affine_grid + grid_sample`.

```python
grid = F.affine_grid(theta, size=target_size, align_corners=False)
G_warp = F.grid_sample(crop_G0, grid, ...)
M_warp = F.grid_sample(crop_M0, grid, ...)
```

---

## 8.2 Mode-conditioned behavior

Let mode indices be:

```python
0 = static
1 = appear/disappear
2 = appearance_change
3 = blink
4 = affine
```

### static

```python
G_mid = G_warp
M_mid = M_warp
```

### appear/disappear

```python
G_mid = 0
M_mid = 0
```

### appearance_change

Keep old appearance:

```python
G_mid = G_warp
M_mid = M_warp
```

### blink

Apply alpha adjustment:

```python
alpha_scale, alpha_bias = split(alpha_params)
G_mid = G_warp * alpha_scale + alpha_bias
M_mid = M_warp
```

### affine

Same renderer, different learned transform:

```python
G_mid = G_warp
M_mid = M_warp
```

So affine is mainly distinguished by the transform head and mode supervision.

---

# 9. Optional OverlayResidualRefiner

You described the main refiner as full-frame after composition, which is good.
But a tiny overlay-level refiner can be useful too.

For v1, I would keep this optional.

If included:

```python
class OverlayResidualRefiner(nn.Module):
    input: [crop_G_mid_base, crop_M_mid, crop_I_bg_local]
    output: delta_G_mid
```

and:

```python
crop_G_mid = crop_G_mid_base + delta_G_mid * crop_M_band
```

But for the first implementation, you can skip this and only do frame-level boundary refinement.

---

# 10. CanvasComposer

This lifts per-instance midpoint overlays back to the full frame.

```python
class CanvasComposer(nn.Module):
    def forward(
        self,
        crop_G_mid,
        crop_M_mid,
        placement,
        I_bg
    ):
        ...
```

## Responsibilities

* paste each object crop back into image canvas
* handle overlap policy
* compose onto `I_bg`

## Output

```python
{
    "G_canvas": FloatTensor[B, 3, H, W],
    "M_canvas": FloatTensor[B, 1, H, W],
    "I_base":   FloatTensor[B, 3, H, W],
}
```

---

## 10.1 Overlap policy

If multiple GO overlap, start with simple masked overwrite or additive residual with clamping.

Recommended v1:

* accumulate `G_canvas`
* accumulate `M_canvas`
* normalize if necessary

Example:

```python
G_canvas += pasted_G
M_canvas = torch.maximum(M_canvas, pasted_M)
I_base = I_bg + G_canvas
```

Since GO residuals are already relative to background, this is reasonable as a first pass.

---

# 11. BoundaryRefiner

This is the final learned cleanup module.

```python
class BoundaryRefiner(nn.Module):
    def forward(
        self,
        I_base,
        I_bg,
        G_canvas,
        M_canvas
    ):
        ...
```

## Goal

Fix only:

* edge blending
* aliasing
* small compositing errors

Not:

* large geometry mistakes
* object identity mistakes

---

## 11.1 Input tensor design

Build a refinement input:

```python
edge = compute_boundary_band(M_canvas)
refine_in = concat([
    I_base,      # 3
    I_bg,        # 3
    G_canvas,    # 3
    M_canvas,    # 1
    edge,        # 1
], dim=1)
```

Total:

```python
11 channels
```

---

## 11.2 Architecture

Use a small UNet-like or shallow encoder-decoder.

Recommended:

```python
class BoundaryRefiner(nn.Module):
    enc1: 11 -> 32
    enc2: 32 -> 64
    enc3: 64 -> 128
    dec2: 128 -> 64
    dec1: 64 -> 32
    out: 32 -> 3
```

Keep it small.

## Output

```python
delta: FloatTensor[B, 3, H, W]
```

Final:

```python
I_final = I_base + delta * edge
```

That mask-gating is important.

---

# 12. Full GOHandlerNet forward

Now the top-level flow is:

```python
def forward(I0, I1, I0_bg, I1_bg, C, I_bg):
    packed = crop_builder(...)
    enc0 = instance_encoder(X0)
    enc1 = instance_encoder(X1)
    interp = temporal_interpreter(enc0, enc1, meta, masks)
    rendered = midpoint_renderer(...)
    composed = canvas_composer(rendered, placement, I_bg)
    refined = boundary_refiner(...)
    return outputs
```

## Recommended outputs dict

```python
{
    # final frame outputs
    "I_base": ...,
    "I_final": ...,

    # canvas-level outputs
    "G_canvas": ...,
    "M_canvas": ...,

    # per-instance outputs
    "crop_G_mid_base": ...,
    "crop_M_mid": ...,
    "mode_logits": ...,
    "presence_logits": ...,
    "transform_params": ...,
    "alpha_params": ...,
    "confidence": ...,

    # bookkeeping
    "instance_batch_idx": ...,
    "instance_boxes_xyxy": ...,
}
```

This is exactly what you want for losses and debugging.

---

# 13. Loss module design

Create a separate loss class.

```python
class GOHandlerLoss(nn.Module):
    def forward(self, outputs, targets):
        ...
```

## Supported targets

```python
targets = {
    "mode_gt": LongTensor[N],
    "transform_gt": FloatTensor[N, 6],      # optional
    "crop_G_mid_gt": FloatTensor[N, 3, Hp, Wp],   # optional if available per-object
    "I_final_gt": FloatTensor[B, 3, H, W],
}
```

## Loss terms

### mode loss

```python
L_mode = cross_entropy(outputs["mode_logits"], targets["mode_gt"])
```

### transform loss

```python
L_transform = smooth_l1(outputs["transform_params"], targets["transform_gt"])
```

### GO midpoint loss

If you have expected GO appearance per object:

```python
L_go = l1(outputs["crop_G_mid_base"], targets["crop_G_mid_gt"])
```

or on refined GO if you later add overlay refiner.

### final frame loss

```python
L_final = l1(outputs["I_final"], targets["I_final_gt"])
```

### optional edge loss

Apply around mask band only.

---

# 14. Concrete block definitions Codex should implement

Here is the exact list of classes I would ask for:

```python
class ConvNormAct(nn.Module)
class ResidualBlock(nn.Module)

class OverlayCropBuilder(nn.Module)          # mostly utility / deterministic
class InstanceEncoder(nn.Module)
class GeometryEncoder(nn.Module)
class CorrespondenceBlock(nn.Module)
class FusionBlock(nn.Module)

class PresenceHead(nn.Module)
class ModeHead(nn.Module)
class TransformHead(nn.Module)
class AlphaHead(nn.Module)
class ConfidenceHead(nn.Module)

class TemporalInterpreter(nn.Module)
class MidpointWarpRenderer(nn.Module)
class CanvasComposer(nn.Module)
class BoundaryRefiner(nn.Module)

class GOHandlerNet(nn.Module)
class GOHandlerLoss(nn.Module)
```

That is the right level of granularity.

---

# 15. What to tell Codex explicitly

The best prompt instruction is:

Implement the GO handler as these explicit PyTorch modules:

* ConvNormAct
* ResidualBlock
* OverlayCropBuilder
* InstanceEncoder
* GeometryEncoder
* CorrespondenceBlock
* FusionBlock
* PresenceHead
* ModeHead
* TransformHead
* AlphaHead
* ConfidenceHead
* TemporalInterpreter
* MidpointWarpRenderer
* CanvasComposer
* BoundaryRefiner
* GOHandlerNet
* GOHandlerLoss

Requirements:

* keep Interpreter, Renderer, and Refiner separate
* midpoint transform must be explicitly predicted and explicitly used in warping
* composition onto I_bg must happen before BoundaryRefiner
* BoundaryRefiner must be mask-gated near GO boundaries
* return a rich outputs dict for training and debugging
* use clear docstrings with tensor shapes
* implement a clean v1 with reasonable placeholders where needed

---

# 16. My honest recommendation

For v1, do **not** ask Codex to implement:

* generalized non-parametric deformation
* TPS
* flow-based GO warp
* multi-layer alpha compositing with complex z-order

Start with:

* affine midpoint transform
* explicit `grid_sample`
* full-frame boundary refiner
* rich outputs + losses

That is already detailed enough to implement, debug, and train.

If you want, I can now do the next step properly: write the **exact Codex prompt that includes all these block definitions and tensor contracts**, ready to paste.

