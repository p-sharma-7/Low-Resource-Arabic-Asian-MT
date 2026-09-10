"""
gpu_utils.py — CUDA enforcement helpers for WMT26 pipeline.

Call require_cuda() at the top of every entry-point so the pipeline
fails immediately with a clear message instead of silently running
on CPU for hours.
"""
import logging
import os
import subprocess
import sys

import torch

logger = logging.getLogger(__name__)


def require_cuda(min_free_gb: float = 10.0) -> int:
    """
    Assert that at least one CUDA GPU is visible and has enough free VRAM.

    Returns the number of available GPUs.
    Raises RuntimeError with a clear fix message if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        _diagnose_and_abort()

    n = torch.cuda.device_count()
    if n == 0:
        _diagnose_and_abort()

    logger.info("CUDA OK — %d GPU(s) available:", n)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        logger.info(
            "  GPU %d: %s  |  VRAM total=%d MB  free=%d MB",
            i, props.name,
            total // 1024**2,
            free  // 1024**2,
        )

    # Warn if the first GPU has less free memory than expected
    free0, _ = torch.cuda.mem_get_info(0)
    free0_gb = free0 / 1024**3
    if free0_gb < min_free_gb:
        logger.warning(
            "GPU 0 has only %.1f GB free (threshold %.1f GB). "
            "Consider clearing other processes first.",
            free0_gb, min_free_gb,
        )

    return n


def get_device_map() -> str:
    """
    Return "cuda" for single-GPU, "auto" for multi-GPU.
    Always errors out (via require_cuda) if no GPU is present.
    """
    require_cuda()
    n = torch.cuda.device_count()
    return "auto" if n > 1 else "cuda:0"


def _diagnose_and_abort() -> None:
    """Print a targeted fix message then abort."""
    msg_lines = [
        "",
        "═" * 64,
        "  ERROR: No CUDA GPU detected by PyTorch.",
        "═" * 64,
        "",
        "  Common causes and fixes:",
        "",
        "  1. Driver/PyTorch version mismatch (most likely):",
        "       Check driver:  nvidia-smi",
        "       Check torch :  python -c \"import torch; print(torch.version.cuda)\"",
        "     Fix — reinstall PyTorch matching your driver CUDA version:",
        "       Driver ≥ 525  → CUDA 12.x",
        "         pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "       Driver ≥ 450  → CUDA 11.x",
        "         pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "",
        "  2. GPU locked by another process:",
        "       nvidia-smi   # look for processes using device memory",
        "       fuser /dev/nvidia*",
        "",
        "  3. Wrong conda/venv active:",
        "       which python && python -c \"import torch; print(torch.__file__)\"",
        "",
        "═" * 64,
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    sys.exit(1)
