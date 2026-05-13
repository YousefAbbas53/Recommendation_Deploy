"""
LITVISION Recommendation API — Utility Module
===============================================
Production logging, device management, CUDA OOM handling,
and temp/cache cleanup helpers.
"""

import os
import gc
import logging
import shutil
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure production-grade structured logging."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )
    # Silence overly chatty third-party loggers
    for noisy in ("transformers", "sentence_transformers", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("litvision.recommendation")

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"CUDA device detected: {gpu_name} ({mem:.1f} GB)")
    else:
        device = "cpu"
        logger.info("No CUDA device — running on CPU")
    return device


def safe_cuda_empty_cache() -> None:
    """Clear CUDA cache if available; silently no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CUDA cache cleared")


def handle_cuda_oom(exc: Exception) -> str:
    """Handle a CUDA OOM exception: clear caches and return a user message."""
    safe_cuda_empty_cache()
    msg = (
        "GPU out of memory during recommendation generation. "
        "The CUDA cache has been cleared. Please retry with a smaller request."
    )
    logger.error(f"CUDA OOM: {exc}")
    return msg

# ---------------------------------------------------------------------------
# Temp / cache cleanup
# ---------------------------------------------------------------------------

_TEMP_DIRS = [
    os.environ.get("HF_HOME", "/tmp/huggingface"),
]


def cleanup_temp_files() -> None:
    """Remove transient cache artefacts that are safe to delete."""
    for d in _TEMP_DIRS:
        cache_dir = os.path.join(d, "hub", ".locks")
        if os.path.isdir(cache_dir):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info(f"Cleaned lock dir: {cache_dir}")
            except Exception as e:
                logger.warning(f"Could not clean {cache_dir}: {e}")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_positive_int(value: int, name: str, max_val: Optional[int] = None) -> int:
    """Ensure *value* is a positive integer, optionally capped at *max_val*."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be ≤ {max_val}, got {value}")
    return value
