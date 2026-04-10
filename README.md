# GOVFI

This repository collects the local pipelines used for GOVFI experiments around:

- dataset preparation and synthetic augmentation
- segmentation and tracked mask extraction
- mask-aware image inpainting
- video frame interpolation
- an in-progress GO-aware model stack under `GONet/`

The repo is a working research workspace rather than a single packaged library. Most workflows are script-driven.

## Repository Layout

- `Datasets/`: dataset sources, builders, augmentation scripts, and inconsistency-generation pipelines
- `GOSeg/`: segmentation and detection inference utilities, including tracked inference
- `MAT/`: local copy of MAT inpainting code and weights directory
- `UPR-Net/`: local copy of UPR-Net frame interpolation code and checkpoints
- `GONet/`: experimental GO-focused model code
- `Test/`: local outputs, scratch runs, and visual inspection artifacts
- `requirements.txt`: top-level environment used across the combined workspace

## Environment Setup

From the repo root:

```bash
cd /var/home/anntynn/Desktop/Skool/GOVFI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Some subprojects also ship their own dependency files:

```bash
pip install -r MAT/requirements.txt
pip install -r UPR-Net/requirements.txt
```

Notes:

- The root requirements assume a CUDA-enabled PyTorch stack.
- `UPR-Net` depends on CuPy for forward warping.
- Exact CUDA and PyTorch compatibility details are documented in [UPR-Net/README.md](/var/home/anntynn/Desktop/Skool/GOVFI/UPR-Net/README.md) and [MAT/README.md](/var/home/anntynn/Desktop/Skool/GOVFI/MAT/README.md).

## Main Work Areas

### Dataset Preparation

Top-level dataset scripts currently in use:

- `Datasets/build_davis_bg.py`
- `Datasets/extract_davis_seq.py`
- `Datasets/build_davis_seq_aug.py`
- `Datasets/build_davis_seq_aug2.py`
- `Datasets/Inconsistencies/build_all.py`
- `Datasets/Inconsistencies/build_dataset.py`
- `Datasets/Inconsistencies/export_views.py`
- `Datasets/Inconsistencies/export_uprnet_hackview.py`
- `Datasets/Inconsistencies/export_external_hackviews.py`

Current dataset trees present in this workspace include:

- `Datasets/DAVIS`
- `Datasets/DAVIS-Triplet`
- `Datasets/DAVIS-Triplet-Aug`
- `Datasets/DAVIS-Septuplet`
- `Datasets/DAVIS-Seq-Aug`
- `Datasets/DAVIS-Seq-Aug-Examine`
- `Datasets/DAVIS-Triplet-Affine-Examine`
- `Datasets/GOoNS`
- `Datasets/SelectedGOoNS`
- `Datasets/Inconsistencies`
- `Datasets/CRAWL`
- `Datasets/vimeo_triplet`

Configuration files live in:

- `Datasets/config.py`
- `Datasets/config2.py`
- `Datasets/Inconsistencies/configs/`

The repository `.gitignore` is set up to keep dataset scripts and lightweight metadata tracked while ignoring bulk dataset payloads and generated archives.

### Segmentation

Primary entry points:

- `GOSeg/inference.py`
- `GOSeg/inference_detection.py`
- `GOSeg/tracked_inference.py`

This area also contains local model artifacts and test material. Weights are intentionally ignored in Git.

### Inpainting

Primary entry points:

- `MAT/generate_image.py`
- `MAT/train.py`

Model weights are expected under:

- `MAT/pretrained/`

See [MAT/README.md](/var/home/anntynn/Desktop/Skool/GOVFI/MAT/README.md) for upstream usage details and training options.

### Frame Interpolation

Primary entry points:

- `UPR-Net/demo/interp_imgs.py`
- `UPR-Net/demo/interp_video.py`
- `UPR-Net/tools/train.py`
- `UPR-Net/tools/benchmark_vimeo90k.py`
- `UPR-Net/tools/benchmark_ucf101.py`
- `UPR-Net/tools/benchmark_snufilm.py`
- `UPR-Net/tools/benchmark_8x_4k1000fps.py`
- `UPR-Net/tools/runtime.py`
- `UPR-Net/run_interp.sh`

Checkpoint files are expected under:

- `UPR-Net/checkpoints/`

See [UPR-Net/README.md](/var/home/anntynn/Desktop/Skool/GOVFI/UPR-Net/README.md) for model-specific commands and benchmark notes.

### GO Model Work

`GONet/` currently contains model modules and design notes for a GO-aware pipeline:

- `GONet/models/`
- `GONet/AGENT_Details.MD`
- `GONet/AGENT_Plan.MD`

This area appears to be under active development rather than a finished CLI workflow.

## Typical Local Workflow

1. Prepare or extract dataset assets under `Datasets/`.
2. Run segmentation or tracking from `GOSeg/` to obtain masks or detections.
3. Use `MAT/` for inpainting where background completion is needed.
4. Use `UPR-Net/` for frame interpolation experiments.
5. Inspect generated outputs under `Test/` or dataset-specific output folders.

## Large Local Artifacts

The repository intentionally ignores large local assets such as:

- dataset payloads under `Datasets/`
- `GOSeg/*.pt` and similar weight files
- `MAT/pretrained/`
- `UPR-Net/checkpoints/`
- local virtual environments such as `venv/`

If a script fails due to missing assets, check those locations first.

## Notes

- `Test/` is a local workspace for outputs and visual inspection artifacts, not a stable benchmark harness.
- Several directories are vendored research codebases with their own README files and assumptions.
- For exact arguments and file contracts, inspect the target script directly before running long jobs.
