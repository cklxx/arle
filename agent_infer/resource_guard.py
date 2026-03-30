"""Resource safety checks for MLX backend on Apple Silicon.

Guards against OOM and disk exhaustion before / during inference.
"""
from __future__ import annotations

import gc
import logging
import platform
import shutil
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

# Limits
_MEMORY_LIMIT_FRACTION = 0.60    # warn when model > 60 % of available RAM
_MEMORY_CRITICAL_FRACTION = 0.90  # trigger GC when process uses > 90 % of total RAM
_MIN_FREE_GB = 2.0
_MAX_CACHE_GB = 20.0


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_apple_silicon() -> None:
    """Raise RuntimeError if not running on Apple Silicon (darwin/arm64)."""
    sys_name = platform.system()
    machine = platform.machine()
    if sys_name != "Darwin" or machine != "arm64":
        raise RuntimeError(
            f"MLX backend requires Apple Silicon (Darwin/arm64). "
            f"Got: {sys_name}/{machine}"
        )


def check_memory(min_free_gb: float = _MIN_FREE_GB) -> None:
    """Raise RuntimeError if less than *min_free_gb* GB of RAM is available."""
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    if avail_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient memory: {avail_gb:.1f} GB available "
            f"(total {total_gb:.1f} GB). Need at least {min_free_gb:.1f} GB free."
        )
    logger.info(
        "Memory check OK: %.1f GB available / %.1f GB total",
        avail_gb, total_gb,
    )


def check_disk(cache_dir: str | Path, max_gb: float = _MAX_CACHE_GB) -> None:
    """Warn and evict oldest top-level entries in *cache_dir* if over *max_gb*."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return

    total_bytes = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    total_gb = total_bytes / (1024 ** 3)

    if total_gb <= max_gb:
        logger.info("Disk check OK: cache %.1f GB / %.1f GB limit", total_gb, max_gb)
        return

    logger.warning(
        "Model cache %.1f GB exceeds %.1f GB limit — evicting oldest entries.",
        total_gb, max_gb,
    )
    entries = sorted(
        list(cache_dir.iterdir()),
        key=lambda e: e.stat().st_mtime,
    )
    target_gb = max_gb * 0.80  # evict until 80 % of limit
    for entry in entries:
        if total_gb <= target_gb:
            break
        entry_gb = (
            sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            if entry.is_dir()
            else entry.stat().st_size
        ) / (1024 ** 3)
        logger.warning("Evicting %s (%.2f GB)", entry, entry_gb)
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        total_gb -= entry_gb


def check_model_fits(model_size_gb: float) -> None:
    """Log a warning when *model_size_gb* exceeds 60 % of available memory."""
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    limit_gb = avail_gb * _MEMORY_LIMIT_FRACTION
    if model_size_gb > limit_gb:
        logger.warning(
            "Model size %.1f GB exceeds 60%% of available memory "
            "(limit %.1f GB, available %.1f GB). "
            "Inference may be unstable.",
            model_size_gb, limit_gb, avail_gb,
        )
    else:
        logger.info(
            "Model size %.1f GB fits within memory limit (%.1f GB).",
            model_size_gb, limit_gb,
        )


# ---------------------------------------------------------------------------
# Periodic in-flight check
# ---------------------------------------------------------------------------

def periodic_memory_check(step: int, interval: int = 100) -> None:
    """Every *interval* decode steps, check RAM usage and trigger GC if critical.

    Call this inside the generation loop::

        for i, response in enumerate(stream_generate(...)):
            periodic_memory_check(i)
            ...
    """
    if step % interval != 0:
        return

    vm = psutil.virtual_memory()
    used_pct = vm.percent  # system-wide percent used

    if used_pct / 100 > _MEMORY_CRITICAL_FRACTION:
        logger.warning(
            "Memory critical at step %d: %.0f%% used "
            "(%.1f / %.1f GB). Running GC.",
            step, used_pct,
            vm.used / (1024 ** 3),
            vm.total / (1024 ** 3),
        )
        gc.collect()
        vm2 = psutil.virtual_memory()
        logger.info("After GC: %.0f%% used.", vm2.percent)
