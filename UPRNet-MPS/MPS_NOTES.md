# UPRNet-MPS Notes

This copy is an inference-oriented UPR-Net variant for non-CUDA environments.

Changes in this copy:
- device selection prefers `cuda`, then `mps`, then `cpu`
- requesting `--device mps` falls back to CPU with a warning when PyTorch does not expose MPS
- CUDA/CuPy-only `softsplat` and `correlation` operators were replaced with pure PyTorch implementations

Current limitation on this machine:
- the active Python environment reports `torch.backends.mps.is_available() == False`
- because of that, `--device mps` cannot use Metal yet and will fall back to CPU

Validated command from this copy:

```bash
PYTHONPATH=. ../.venv/bin/python demo/interp_imgs.py \
  --frame0 demo/images/beanbags0.png \
  --frame1 demo/images/beanbags1.png \
  --save_dir demo/output_mps_requested_smoke \
  --device mps
```

