"""
GPU memory helpers for a 6 GB card.

The pipeline is sequential on purpose: speech model, then lip sync, then (in
the studio) a diffusion model, never two heavy ones at once. These helpers are
how each stage makes room for the next and refuses to start when it cannot --
a clear "not enough VRAM, here is who is holding it" before loading, instead
of a CUDA out-of-memory traceback halfway through a render.

torch is imported lazily so modules that only need the *check* (and the unit
tests) do not pay for importing it.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Callables that drop a module's cached models. Engines register themselves so
# "free the GPU" does not need to know what is loaded.
_releasers: Dict[str, Callable[[], None]] = {}


class InsufficientVRAM(RuntimeError):
    """Not enough free GPU memory to load a model."""


def register_releaser(name: str, release: Callable[[], None]) -> None:
    """Register a function that unloads ``name``'s models from the GPU."""
    _releasers[name] = release


def preferred_device() -> Optional[str]:
    """
    ``AVATAR_DEVICE`` ("cpu" or "cuda") when set, else ``None`` for automatic.

    The escape hatch for a GPU someone else is using: every vision model here
    is small enough to run on CPU, just slower.
    """
    import os

    value = os.getenv("AVATAR_DEVICE", "").strip().lower()
    return value if value in ("cpu", "cuda") else None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def free_vram_mb() -> Optional[int]:
    """Free GPU memory in MiB, or ``None`` without CUDA."""
    if not cuda_available():
        return None
    import torch

    free, _total = torch.cuda.mem_get_info()
    return int(free // (1024 * 1024))


def empty_cache() -> None:
    """Return cached, unused blocks to the driver."""
    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()


def release_others(keep: Optional[str] = None) -> List[str]:
    """
    Unload every registered model except ``keep``.

    Returns the names released. A releaser that raises is logged and skipped:
    failing to free one model must not stop the others being freed.
    """
    released: List[str] = []
    for name, release in list(_releasers.items()):
        if name == keep:
            continue
        try:
            release()
            released.append(name)
        except Exception as err:  # noqa: BLE001
            logger.warning("Could not release %s from the GPU: %s", name, err)
    empty_cache()
    return released


def ensure_vram(required_mb: int, purpose: str, keep: Optional[str] = None) -> None:
    """
    Make sure ``required_mb`` of GPU memory is free before loading a model.

    First tries to get there by unloading the other registered models. If it
    still is not enough -- another process holds the card -- raise with the
    numbers, so the fix (close the other process) is obvious.

    A machine without CUDA passes: the caller runs on CPU and this is not the
    check that applies.
    """
    free = free_vram_mb()
    if free is None or free >= required_mb:
        return
    released = release_others(keep=keep)
    free = free_vram_mb() or 0
    if free >= required_mb:
        logger.info(
            "Freed GPU memory for %s by unloading: %s", purpose, ", ".join(released) or "cache"
        )
        return
    import torch

    own = int(torch.cuda.memory_allocated() // (1024 * 1024))
    holder = (
        f"This process still holds {own} MiB in models that did not register a "
        "releaser; restart it, or render in a separate process."
        if own > 256
        else "Another process is holding the card: check `nvidia-smi`, close it, and retry."
    )
    raise InsufficientVRAM(
        f"{purpose} needs about {required_mb} MiB of free GPU memory but only "
        f"{free} MiB is free. {holder}"
    )


def reset_peak() -> None:
    if cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> Optional[int]:
    """Peak GPU memory allocated by this process since ``reset_peak``."""
    if not cuda_available():
        return None
    import torch

    return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
