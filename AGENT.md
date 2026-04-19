# AGENT.md

This file defines how an agent should operate in this repository.

## Repo Nature

`GO-VFI` is a script-driven research workspace, not a single clean package.

It combines several work areas:

- `Datasets/`: dataset extraction, augmentation, and inconsistency generation
- `GOSeg/`: segmentation, detection, and tracked inference
- `MAT/`: inpainting code and pretrained weights
- `UPR-Net/`: frame interpolation code, demos, and training tools
- `GONet/`: in-progress GO-aware model work

Assume that:

- multiple subdirectories are vendored research code with their own conventions
- scripts are the main interface, not reusable APIs
- local data and weight artifacts are large and mostly ignored by Git
- exact behavior often lives in the target script, not in a centralized framework

## Environment Standard

Use the repo-local virtual environment:

- interpreter: `.venv/bin/python`
- packages: root `requirements.txt`

Do not install or setup globally with bare `python`, use `.venv/bin/python ...` when running instead.

## Device Principle

This repo should support a split workflow:

- local development on macOS with `cpu` or `mps`
- real training or throughput-sensitive runs on a remote Linux server with `cuda` GPUs such as A100

Do not hardcode `cuda` into general training logic.

The correct rule is:

- keep the default codepath backend-agnostic
- isolate backend-specific choices in a small runtime boundary
- allow CUDA-specific optimizations without forcing the whole repo to depend on CUDA

## Device Policy

Backend selection should happen in one place:

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Use `.to(device)` consistently for:

- models
- inputs
- targets
- persistent tensors created by the training loop

Avoid:

- scattered `"cuda"` string literals
- `.cuda()` calls embedded in model code
- backend branches spread across unrelated files

## Precision Policy

Treat AMP and dtype as runtime policy, not model logic.

- Prefer AMP on `cuda`
- Keep a correct non-AMP path for `mps` and `cpu`
- Do not assume `mps` has feature parity with CUDA-oriented training stacks

Example:

```python
use_amp = device.type == "cuda"

if use_amp:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(inputs, targets)
else:
    loss = model(inputs, targets)
```

## Local vs Remote Expectations

Use local macOS runs for:

- shape checks
- script validation
- dataset pipeline validation
- overfit-one-batch tests
- small smoke tests
- quick timing experiments

Use the A100 server for:

- real training runs
- throughput tuning
- mixed precision
- large batch sizes
- distributed training
- final benchmark claims

Do not infer A100 training behavior from local `mps` timing.

## Repo-Specific Working Rules

### 1. Read the target script first

Before modifying or running a workflow, inspect the actual entrypoint script.

Important examples:

- `GOSeg/inference.py`
- `GOSeg/inference_detection.py`
- `GOSeg/tracked_inference.py`
- `MAT/generate_image.py`
- `MAT/train.py`
- `UPR-Net/tools/train.py`
- `UPR-Net/tools/runtime.py`
- `UPR-Net/demo/interp_imgs.py`
- `UPR-Net/demo/interp_video.py`

This repo does not have a single unified CLI contract.

### 2. Respect vendored subprojects

`MAT/` and `UPR-Net/` are not cleanly designed around this repo’s top-level conventions.

When editing them:

- make the smallest viable change
- preserve upstream structure unless a repo-specific adaptation is necessary
- avoid broad refactors unless the user explicitly asks for them

### 3. Do not assume assets exist

Many workflows require local weights or datasets that are intentionally not tracked.

Common missing-asset locations:

- `MAT/pretrained/`
- `UPR-Net/checkpoints/`
- large trees under `Datasets/`
- local model files under `GOSeg/`

If a run fails because an asset is missing, report the missing path clearly instead of guessing.

### 4. Keep generated junk out of the repo

Do not leave behind benchmark scripts, debug dumps, large outputs, or ad hoc artifacts unless the user asked to keep them.

Prefer temporary scripts only when needed, and clean them up after the experiment.

### 5. Treat `Test/` as scratch space

`Test/` is for local outputs and visual inspection artifacts, not a stable benchmark harness.

If you need a quick output location, `Test/` is acceptable. Do not present it as production infrastructure.

## Runtime Boundary Pattern

If you need backend-aware training code, keep it behind a tiny runtime object:

```python
from dataclasses import dataclass
import torch

@dataclass
class Runtime:
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype | None

def make_runtime():
    if torch.cuda.is_available():
        return Runtime(torch.device("cuda"), True, torch.bfloat16)
    if torch.backends.mps.is_available():
        return Runtime(torch.device("mps"), False, None)
    return Runtime(torch.device("cpu"), False, None)
```

The rest of the training code should depend on `Runtime`, not on backend names.

## Anti-Patterns In This Repo

Avoid:

- hardcoding `"cuda"` across dataset, training, and inference scripts
- forcing local Mac development to depend on CUDA-only packages
- rewriting large parts of vendored code when a narrow patch would work
- trusting local performance as evidence for remote server throughput
- modifying ignored datasets or weight directories unless the task requires it

## Decision Standard

For this repository, a good agent change has these properties:

- it works with the repo-local `.venv`
- it does not break macOS local iteration
- it does not block later CUDA execution on the A100 server
- it keeps backend-specific logic narrow and explicit
- it avoids unnecessary churn in vendored subprojects

Default to portability, small patches, and script-level clarity.
