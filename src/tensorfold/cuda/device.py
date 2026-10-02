"""The GPU this process runs on: device 0, or --device / TF_CUDA_DEVICE.

Two ranks on one machine both see both GPUs (CUDA_VISIBLE_DEVICES=0,1) and pick their own here, so NCCL can map the
peer over NVLink; with one GPU visible a rank cannot, and the transfers fall back to host memory."""

from __future__ import annotations

import os


def index() -> int:
    """This process's CUDA device: TF_CUDA_DEVICE (set by ``--device``), else 0."""

    value = os.environ.get("TF_CUDA_DEVICE", "").strip()
    return int(value) if value else 0


def select(torch) -> int:
    """Make ``index()`` the current device (every ``"cuda"`` tensor after this lands there) and return it."""

    i = index()
    if i >= torch.cuda.device_count():
        raise ValueError(f"--device {i}, but {torch.cuda.device_count()} CUDA device(s) are visible "
                         "(CUDA_VISIBLE_DEVICES)")
    torch.cuda.set_device(i)
    return i
