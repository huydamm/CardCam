#!/usr/bin/env bash
# Run train_model.py on your RTX 4060 via WSL2 + tensorflow[and-cuda]
#
# From Windows Terminal:
#   wsl -d Ubuntu -- bash /mnt/c/Users/Asus/Documents/pokercam/train_gpu.sh
#
# From inside WSL:
#   bash /mnt/c/Users/Asus/Documents/pokercam/train_gpu.sh

NV="$HOME/tf-gpu/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cuda_cupti/lib:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$HOME/tf-gpu/bin/activate"
cd /mnt/c/Users/Asus/Documents/pokercam
python3 train_model.py
