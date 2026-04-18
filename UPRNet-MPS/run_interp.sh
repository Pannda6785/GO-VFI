#!/usr/bin/env bash
set -euo pipefail

VENV_ROOT="/var/home/anntynn/Desktop/Skool/SC203-SciMethod/venv"
NVIDIA_SITE_PKGS="$VENV_ROOT/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH="$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/curand/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/cusparselt/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvshmem/lib:$NVIDIA_SITE_PKGS/nvtx/lib:${LD_LIBRARY_PATH:-}"

exec "$VENV_ROOT/bin/python3" -m demo.interp_imgs "$@"
