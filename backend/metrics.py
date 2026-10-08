"""
Operational metrics for ``GET /api/v1/metrics`` (R-51): what the queue is doing and how the renders
and their scores have turned out.

Everything here is computed on request from data the service already keeps (the queue's jobs and
their results); nothing is sampled in the background and nothing is invented. Where there is nothing
to report (no render has finished, no clip has been scored) the field is ``None``/0, never a made-up
number. Only the in-process queue can list its jobs; the Celery backend cannot, and says so.

Where it sits: ``app.py`` calls :func:`summarise` with the in-process queue's job list and merges
the result into the metrics response. Nothing else depends on this module.

Concepts used here:
  * A *percentile* p95 is the value that 95% of observations are at or below. It shows the slow
    tail that a mean hides: one 60 s render among nineteen 5 s renders barely moves the mean but
    sets the p95.
  * *Nearest-rank* is one way to pick that value: sort, then take the item at position
    ``ceil(q * n)`` (1-based). Unlike interpolating methods it always returns a real observation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


def spread(values: List[float]) -> Dict[str, Any]:
    """Count, mean, median, 95th percentile and max of a list; ``None`` for each when the list is empty.

    The percentile is nearest-rank (the smallest value with at least 95% of the data at or below it),
    which never reports a number that was not observed.
    """
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def rank(q: float) -> float:
        """Nearest-rank percentile ``q`` (0..1) of ``ordered``."""
        # ceil(q * n) is the 1-based rank; "- 1" turns it into a list index. max(0, ...) guards
        # q = 0, where ceil gives 0 and the index would wrap round to the last element.
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    # Each figure is rounded to 3 decimals; ``max`` is simply the last item of the sorted list.

    return {"n": len(ordered), "mean": round(sum(ordered) / len(ordered), 3),
            "p50": round(rank(0.5), 3), "p95": round(rank(0.95), 3), "max": round(ordered[-1], 3)}


def summarise(jobs: Iterable[Any]) -> Dict[str, Any]:
    """Queue depth, render timings and lip-sync scores over ``QueuedJob`` objects.

    ``jobs`` is any iterable of objects with ``.status`` (an enum) and ``.result`` (the finished
    job's dict, or ``None``). Returns a JSON-ready dict with ``queue``, ``renders`` and ``quality``.
    One pass over the jobs fills plain lists, which :func:`spread` then summarises.
    """
    states: Dict[str, int] = {}
    engines: Dict[str, int] = {}
    render_s: List[float] = []
    realtime: List[float] = []
    vram: List[float] = []
    marked = unmarked = with_warnings = finished = 0
    lse_c: List[float] = []
    lse_d: List[float] = []
    offsets: List[int] = []
    for queued in jobs:
        # Every job counts toward queue depth, finished or not.
        states[queued.status.value] = states.get(queued.status.value, 0) + 1
        result: Optional[Dict[str, Any]] = queued.result
        if not result:
            # Queued, running or failed: no render numbers to add.
            continue
        engines[result.get("engine") or "?"] = engines.get(result.get("engine") or "?", 0) + 1
        # A result from an older build may lack a field; skip it rather than fail the whole report.
        finished += 1
        if result.get("renderSeconds") is not None:
            render_s.append(float(result["renderSeconds"]))
        if result.get("realtimeFactor") is not None:
            realtime.append(float(result["realtimeFactor"]))
        if result.get("peakVramMb") is not None:
            vram.append(float(result["peakVramMb"]))
        if (result.get("watermark") or {}).get("applied"):
            marked += 1
        else:
            unmarked += 1
        # bool is a subclass of int in Python, so True adds 1 and False adds 0.
        with_warnings += bool(result.get("warnings"))
        # LSE-C / LSE-D are SyncNet lip-sync scores (see lipsync_metric.py); present only when a
        # client asked for the clip to be scored.
        score = result.get("lipsync")
        if score:
            lse_c.append(float(score["lseC"]))
            lse_d.append(float(score["lseD"]))
            offsets.append(int(score["offsetFrames"]))
    return {
        "queue": {"jobsByState": states, "waiting": states.get("QUEUED", 0), "running": states.get("PROCESSING", 0)},
        "renders": {
            "finished": finished, "byEngine": engines, "renderSeconds": spread(render_s),
            # realtimeFactor = render time / video length, so 1.0 is real time and below 1 is faster.
            "realtimeFactor": spread(realtime),
            "peakVramMb": spread(vram),  # only present when the render ran on a GPU
            "watermarked": marked, "notWatermarked": unmarked, "withWarnings": with_warnings,
        },
        "quality": {
            "lipSyncScored": len(offsets), "lseC": spread(lse_c), "lseD": spread(lse_d),
            # Clips whose best audio/video offset is -1, 0 or +1 frame: in sync to the eye.
            "offsetWithinOneFrame": sum(1 for o in offsets if abs(o) <= 1),
            "note": "only renders someone scored via POST .../lipsync-score appear here",
        },
    }
