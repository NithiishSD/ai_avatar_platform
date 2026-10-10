"""
GPU memory helpers for a 6 GB card.

The pipeline is sequential on purpose: speech model, then lip sync, then (in
the studio) a diffusion model, never two heavy ones at once. These helpers are
how each stage makes room for the next and refuses to start when it cannot --
a clear "not enough VRAM, here is who is holding it" before loading, instead
of a CUDA out-of-memory traceback halfway through a render.

torch is imported lazily so modules that only need the *check* (and the unit
tests) do not pay for importing it.

The design worth noticing: this module never knows what models exist. Each
engine *registers* a callback that unloads its own models, so "free the GPU"
is a message broadcast to whoever is listening. Adding a sixth model does not
change this file. That is the observer / registry pattern, and the alternative
(an if-chain naming every engine here) would couple this module to all of them.

**How to say this in an interview:** "Memory pressure is handled with a
registry of release callbacks rather than central knowledge of every model, so
the guard stays closed to modification as engines are added."
"""

# Postpones evaluation of annotations, so a type can be written as `List[str]`
# or reference a class defined later in the file without quoting it. Standard
# at the top of typed modules on Python 3.7-3.9; harmless and still useful here.
from __future__ import annotations

import gc
import logging
from typing import Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Callables that drop a module's cached models. Engines register themselves so
# "free the GPU" does not need to know what is loaded.
#
# Callable[[], None] = "a function taking no arguments and returning nothing".
_releasers: Dict[str, Callable[[], None]] = {}


class InsufficientVRAM(RuntimeError):
    """
    Not enough free GPU memory to load a model.

    A *custom exception type*, so callers can catch exactly this and fall back
    to CPU, rather than catching RuntimeError and accidentally swallowing
    unrelated bugs. Subclassing RuntimeError (not Exception) keeps it in the
    right family for anything doing broad handling.
    """


def register_releaser(name: str, release: Callable[[], None]) -> None:
    """Register a function that unloads ``name``'s models from the GPU."""
    # Keyed by name, so registering twice replaces rather than duplicates -
    # which matters because a module re-imported under test would otherwise
    # stack up stale releasers pointing at dead objects.
    _releasers[name] = release


def preferred_device() -> Optional[str]:
    """
    ``AVATAR_DEVICE`` ("cpu" or "cuda") when set, else ``None`` for automatic.

    The escape hatch for a GPU someone else is using: every vision model here
    is small enough to run on CPU, just slower.
    """
    import os

    # Three-valued on purpose: "cpu" and "cuda" are explicit choices, None
    # means "decide for me". Returning "cuda" as a default would remove the
    # caller's ability to tell "unset" from "explicitly asked for CUDA".
    value = os.getenv("AVATAR_DEVICE", "").strip().lower()
    # Anything unrecognised (a typo like "gpu") is treated as unset rather than
    # raising: a bad env var should not stop the process booting.
    return value if value in ("cpu", "cuda") else None


def cuda_available() -> bool:
    """True when torch can actually see a GPU."""
    try:
        import torch

        # bool() because torch returns a numpy bool in some versions, which is
        # truthy but not `is True` - and callers compare it.
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        # Broad on purpose: torch may be missing, or present but unable to
        # initialise its driver. Either way the answer is "no GPU", and a
        # capability probe must never be the thing that crashes the process.
        return False


def free_vram_mb() -> Optional[int]:
    """Free GPU memory in MiB, or ``None`` without CUDA."""
    if not cuda_available():
        # None, not 0: "unknown because there is no GPU" is different from
        # "a GPU with nothing free", and ensure_vram relies on the difference.
        return None
    import torch

    # mem_get_info asks the driver, so it counts memory held by *other*
    # processes too. torch.cuda.memory_allocated() would only see our own
    # tensors and would happily report plenty free while another process
    # occupies the card.
    free, _total = torch.cuda.mem_get_info()
    # The leading underscore marks a deliberately unused value.
    # // is integer division; 1024*1024 converts bytes to MiB.
    return int(free // (1024 * 1024))


class InsufficientRAM(RuntimeError):
    """
    Not enough free system memory to load a model on the CPU.

    The CPU twin of ``InsufficientVRAM``. Without it the failure is the
    operating system's out-of-memory killer ending the whole server mid-request
    (exit 137, no message), which is what happened when several engines were
    loaded one after another into the same process.
    """


# Free memory to keep beyond a model's own size, so loading does not leave the
# machine with nothing for the rest of the request (audio, the aligner, the OS).
RAM_HEADROOM_MB = 1500


def free_ram_mb() -> Optional[int]:
    """Memory the kernel says is available (``MemAvailable``), or ``None`` if it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024  # the file is in kB
    except (OSError, ValueError):
        pass
    # Not Linux, or unreadable: unknown. Callers treat None as "do not interfere".
    return None


def ensure_host_memory(required_mb: int, purpose: str, keep: Optional[str] = None) -> None:
    """
    Make room in system RAM before loading a model that will run on the CPU.

    Golden rule 5 ("one heavy model resident at a time") was only enforced for
    GPU memory; on a CPU host every engine stayed loaded forever. Here, when
    free RAM would fall below the model's size plus ``RAM_HEADROOM_MB``, the
    other registered models are unloaded first (they reload lazily on next
    use). If that still is not enough the load is refused with the numbers,
    rather than left for the OS to kill the process.

    A model that is going to the GPU does not use system RAM for its weights
    in the same way, so this passes when CUDA is in use.
    """
    if cuda_available():
        return
    free = free_ram_mb()
    if free is None or free >= required_mb + RAM_HEADROOM_MB:
        return
    released = release_others(keep=keep)
    free = free_ram_mb() or 0
    if free >= required_mb:
        logger.info("Freed RAM for %s by unloading: %s (now %d MiB free)", purpose, ", ".join(released) or "cache", free)
        return
    raise InsufficientRAM(
        f"{purpose} needs about {required_mb} MiB of free RAM but only {free} MiB is available "
        f"after unloading {', '.join(released) or 'nothing else'}. Close other programs "
        "(check `ps aux --sort=-rss | head`), or run it on the GPU."
    )


def _return_heap_to_os() -> None:
    """
    Ask glibc to give freed heap pages back to the operating system.

    Dropping a model and running the garbage collector frees its objects, but
    glibc keeps the freed pages in the process, so the server's memory stays
    high and the OS (and the next model) cannot use them. Measured 8 Oct 2026
    on one process loading Kokoro, XTTS-v2, OpenVoice and Bark in turn, each
    released: 1.2-2.4 GiB stayed resident after ``gc.collect()``, ~1.0 GiB
    after ``malloc_trim``. Only exists on Linux/glibc; anywhere else this is a no-op.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def empty_cache() -> None:
    """Return cached, unused blocks to the driver."""
    # gc.collect() first, and that order matters: torch only releases a block
    # once no Python object references the tensor, so collecting unreachable
    # objects is what makes the following empty_cache() actually free anything.
    gc.collect()
    _return_heap_to_os()
    if cuda_available():
        import torch

        # torch keeps freed blocks in its own allocator pool for speed. This
        # hands them back to the driver so another process can use them.
        torch.cuda.empty_cache()


def release_others(keep: Optional[str] = None) -> List[str]:
    """
    Unload every registered model except ``keep``.

    Returns the names released. A releaser that raises is logged and skipped:
    failing to free one model must not stop the others being freed.
    """
    released: List[str] = []
    # list(...) copies the items before iterating: a releaser is free to
    # register or drop entries, and mutating a dict while looping over it
    # raises RuntimeError. Cheap insurance for a handful of entries.
    for name, release in list(_releasers.items()):
        if name == keep:
            continue
        try:
            release()
            released.append(name)
        except Exception as err:  # noqa: BLE001
            # Keep going. This is the "best effort cleanup" pattern: the goal
            # is to free as much as possible, so one broken releaser must not
            # abort the loop and strand the remaining models in memory.
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

    Returns nothing and raises on failure, rather than returning a bool. A
    bool invites being ignored; an exception cannot be.
    """
    free = free_vram_mb()
    # `free is None` is the no-CUDA case and passes: see the docstring. Writing
    # this as `if not free` would be a bug - 0 MiB free is falsy but is exactly
    # the situation this guard exists for.
    if free is None or free >= required_mb:
        return
    released = release_others(keep=keep)
    # Re-measure rather than assume the release worked. `or 0` covers the
    # (impossible here, but cheap) None case so the comparison below is safe.
    free = free_vram_mb() or 0
    if free >= required_mb:
        logger.info(
            # `or "cache"` so the message reads sensibly when nothing was
            # registered and the win came from empty_cache() alone.
            "Freed GPU memory for %s by unloading: %s", purpose, ", ".join(released) or "cache"
        )
        return
    import torch

    # memory_allocated() is *our* tensors only. Comparing it against the
    # shortfall is what distinguishes the two causes below.
    own = int(torch.cuda.memory_allocated() // (1024 * 1024))
    # Golden rule 7: the error names the fix. Which fix depends on who is
    # actually holding the memory, so the message is chosen, not generic.
    # 256 MiB is a threshold, not a measurement: below that our own footprint
    # is CUDA context overhead rather than a model nobody released.
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
    """Zero the high-water mark, to measure one stage's peak in isolation."""
    if cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> Optional[int]:
    """Peak GPU memory allocated by this process since ``reset_peak``."""
    if not cuda_available():
        return None
    import torch

    # max_memory_allocated is a high-water mark, not a current reading: it
    # survives the tensors being freed, which is what makes it usable as
    # evidence that a render stayed under 6 GB (the Gate 2 criterion).
    return int(torch.cuda.max_memory_allocated() // (1024 * 1024))


def is_cuda_oom(err: BaseException) -> bool:
    """Whether ``err`` is the GPU running out of memory (PyTorch's own error, or cuBLAS failing to allocate)."""
    text = str(err)
    return type(err).__name__ == "OutOfMemoryError" or ("CUDA" in text and ("out of memory" in text or "ALLOC_FAILED" in text))


def retry_after_freeing(work: Callable[[], T], keep: Optional[str] = None) -> T:
    """
    Run ``work``; if the GPU is out of memory, unload the other registered models and try once more.

    Small models (the watermarks) do not reserve memory up front the way the big ones do, so when a big
    model is still resident they can fail mid-call. One retry after freeing is enough: if it fails
    again, the card really is full and the error is raised as it is.
    """
    try:
        return work()
    except Exception as err:  # noqa: BLE001 - re-raised unless it is a GPU out-of-memory
        if not is_cuda_oom(err):
            raise
        released = release_others(keep=keep)
        empty_cache()
        logger.warning("GPU out of memory for %s; unloaded %s and retrying once", keep or "a model", released or "nothing")
        return work()
