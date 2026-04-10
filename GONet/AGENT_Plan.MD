You are implementing the GO handler branch inside a larger VFI pipeline.

First, do not code yet. Read the repo, inspect the relevant files, and then produce:

1. a concise understanding of the current codebase structure relevant to this task,
2. a proposed implementation plan,
3. the list of files you will create or modify,
4. any ambiguities or missing assumptions that must be fixed before coding.

After that, wait for my approval before writing code.

Task context

* This module is the graphical overlay (GO) handler inside a larger video frame interpolation system.
* Upstream, I already have:

  * object-level GO masks and correspondences between frame 0 and frame 1,
  * inpainted background frames I0_bg and I1_bg,
  * a base VFI output I_bg, which is the interpolated background without GO.
* The GO handler takes:

  * I0, I1,
  * I0_bg, I1_bg,
  * C = set of GO correspondences,
  * I_bg,
    and outputs:
  * I_final, the final interpolated frame with GO restored.

Architecture to implement

* The GO handler has three stages:

  1. Interpreter
  2. Renderer
  3. Refiner

Detailed design

1. Overlay isolation

* For each GO instance i, derive overlay-only residuals:

  * G0_i = (I0 - I0_bg) * M0_i
  * G1_i = (I1 - I1_bg) * M1_i
* The GO branch should reason mainly about overlay appearance and transform, not natural background motion.

2. Interpreter

* Per GO instance, learn:

  * temporal mode,
  * midpoint transform T_i^(0->0.5),
  * optional alpha / visibility parameters.
* Temporal modes:

  * static
  * appear/disappear
  * appearance change
  * blink
  * affine
* The interpreter should use overlay-centric crops and masks.
* I can provide temporal mode GT as supervision.
* I can also provide transform detail GT as supervision, so design the code to support transform loss.

3. Renderer

* The renderer should be structured, not a fully generative decoder.
* It should warp the source GO non-parametrically according to the learned midpoint transform.
* Mode-conditioned behavior:

  * static: warp old GO to midpoint
  * appearance change: keep old appearance, warp to midpoint
  * blink: warp old appearance, then apply alpha adjustment
  * affine: warp with affine midpoint transform
  * appear/disappear: output empty / zero GO
* The renderer should output a midpoint GO layer G_mid.

4. Composition

* First compose:

  * I_base = I_bg + G_mid
    or alpha-aware equivalent if the implementation supports alpha explicitly.

5. Refiner

* The refiner is mainly for boundary cleanup after composition.
* It takes:

  * I_base,
  * I_bg,
  * G_mid,
  * GO masks or midpoint masks,
  * edge-focused mask prior if useful.
* The refiner should mainly fix:

  * edge blending,
  * anti-aliasing,
  * slight alpha mismatch,
  * small local artifacts near GO boundaries.
* It should not be responsible for large motion correction or re-rendering the whole overlay.
* Prefer mask-gated refinement so edits are concentrated near the GO boundary.

Supervision

* temporal mode GT for each overlay,
* expected midpoint GO appearance,
* final full-frame GT I_0.5.

Implementation requirements

* Write this as a modular PyTorch design.
* Separate modules clearly:

  * OverlayExtractor
  * SiameseEncoder or equivalent
  * TemporalInterpreter
  * WarpRenderer
  * EdgeRefiner
  * GOHandlerNet
* Keep the code extensible for multiple GO instances per frame.
* Make the first implementation clean and minimal, but not toy-like.
* Use clear tensor shapes and docstrings.
* Add TODO markers only where a true design choice remains unresolved.
* Do not introduce unnecessary framework complexity.

What I want in the first coding pass

* A clean architecture skeleton in PyTorch
* Forward pass wiring for the full GO handler
* Reasonable placeholder implementations for:

  * transform prediction,
  * warping,
  * edge-mask computation,
  * composition,
  * outputs dictionary for losses
* A loss stub that supports:

  * mode loss
  * transform loss
  * GO midpoint loss
  * final frame loss

Important constraints

* Keep the implementation faithful to the architecture above.
* Do not collapse Interpreter, Renderer, and Refiner into one monolithic UNet.
* Do not make the Refiner responsible for recovering failures that belong to the Interpreter or Renderer.
* Prefer explicit geometry + explicit composition + local refinement.

Before coding, summarize your understanding of the architecture in your own words and point out any mismatch between the requested design and the current repository.

