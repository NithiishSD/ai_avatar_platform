#!/usr/bin/env python3
"""
Score rendered videos against their source photo: LPIPS (N-12) and SFace identity similarity (N-13).

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_visual.py \\
        --video outputs/renders/t52-live.mp4 --face demo --other-face gen-live

``--other-face`` is the negative control: a different registered face, scored against the same
source so the identity number can be read next to what "a different person" scores. Writes the
report to ``outputs/benchmarks/visual_<video>.json`` and prints it.

Concepts used here:

**LPIPS** (Learned Perceptual Image Patch Similarity) is a distance between two images computed
from the features of a trained network. Lower means the frames look more like the photo to a
person, which pixel-by-pixel error does not capture well.

**SFace** is a face-recognition model that turns a face into an identity vector. Comparing the
source photo's vector with each frame's says whether the animated face is still the same person.

**A negative control** is a measurement where the answer is known to be "no". Scoring a different
person shows what a non-match looks like on this scale, so the real score has something to be
read against.

Exit code 0 when every video was scored, 1 when any could not be (the reason is printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# The repository root, two levels up from scripts/measure_visual.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Makes the backend modules importable by name without setting PYTHONPATH.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def main(argv: Optional[List[str]] = None) -> int:
    """
    Score each video and write one JSON report per video.

    ``argv`` defaults to the real command line; tests can pass a list instead.
    """
    # The module docstring doubles as --help text; RawDescriptionHelpFormatter keeps its layout.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, nargs="+", help="one or more rendered MP4s of --face")
    parser.add_argument("--face", required=True, help="avatarId the videos were rendered from")
    parser.add_argument("--other-face", default=None, help="a different avatarId, as the negative control")
    # Scoring every frame is slow and adjacent frames are nearly identical, so sampling loses little.
    parser.add_argument("--every", type=int, default=5, help="score every Nth frame")
    args = parser.parse_args(argv)

    # Late imports: --help stays fast and does not load the vision models.
    import avatar_store
    import visual_metrics

    faces = avatar_store.AvatarStore()
    # decode_image returns the RGB pixels and a second value this script does not need.
    source, _ = avatar_store.decode_image(faces.get(args.face).path)
    other = None
    if args.other_face:
        other, _ = avatar_store.decode_image(faces.get(args.other_face).path)

    # One shared scorer, so the models load once for all the videos.
    scorer = visual_metrics.shared_scorer()
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Starts as success; any video that fails flips it to 1 but the loop carries on with the rest.
    status = 0
    for video in args.video:
        try:
            report = scorer.score_video(video, source, other_person_rgb=other, every=args.every)
        except visual_metrics.VisualMetricError as err:
            print(f"error: {video}: {err}", file=sys.stderr)
            status = 1
            continue
        # Record what was scored and when, in UTC, so a saved number can be traced to its run.
        report.update({"video": str(video), "face": args.face, "date": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        (out_dir / f"visual_{Path(video).stem}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
