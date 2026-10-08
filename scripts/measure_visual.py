#!/usr/bin/env python3
"""
Score rendered videos against their source photo: LPIPS (N-12) and SFace identity similarity (N-13).

    PYTHONPATH=backend backend/.conda/bin/python scripts/measure_visual.py \\
        --video outputs/renders/t52-live.mp4 --face demo --other-face gen-live

``--other-face`` is the negative control: a different registered face, scored against the same
source so the identity number can be read next to what "a different person" scores. Writes the
report to ``outputs/benchmarks/visual_<video>.json`` and prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, nargs="+", help="one or more rendered MP4s of --face")
    parser.add_argument("--face", required=True, help="avatarId the videos were rendered from")
    parser.add_argument("--other-face", default=None, help="a different avatarId, as the negative control")
    parser.add_argument("--every", type=int, default=5, help="score every Nth frame")
    args = parser.parse_args(argv)

    import avatar_store
    import visual_metrics

    faces = avatar_store.AvatarStore()
    source, _ = avatar_store.decode_image(faces.get(args.face).path)
    other = None
    if args.other_face:
        other, _ = avatar_store.decode_image(faces.get(args.other_face).path)

    scorer = visual_metrics.shared_scorer()
    out_dir = PROJECT_ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for video in args.video:
        try:
            report = scorer.score_video(video, source, other_person_rgb=other, every=args.every)
        except visual_metrics.VisualMetricError as err:
            print(f"error: {video}: {err}", file=sys.stderr)
            status = 1
            continue
        report.update({"video": str(video), "face": args.face, "date": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        (out_dir / f"visual_{Path(video).stem}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
