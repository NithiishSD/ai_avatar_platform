#!/usr/bin/env python
"""
Register an avatar face in ``inputs/faces/`` with its provenance record.

Two ways in, matching the two kinds of face the consent rule allows:

  # A face that belongs to nobody (Stable Diffusion 1.5, reproducible by seed)
  PYTHONPATH=backend backend/.conda/bin/python scripts/make_avatar.py \\
      --synthetic --avatar-id demo --seed 7

  # A real person's photo, with the basis on which it may be animated
  PYTHONPATH=backend backend/.conda/bin/python scripts/make_avatar.py \\
      --human photo.jpg --avatar-id nithiish --subject "Full Name" \\
      --consent subject-provided --licence "own photo"

Other commands:
  --list             show every image in the store and whether it is usable
  --delete ID        remove an avatar and its provenance record

Every image goes through the quality gate (one face, facing the camera, large
enough). A photo of a real person is refused without --subject and --consent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import provenance  # noqa: E402
from avatar_store import AvatarError, AvatarNotFound, AvatarRejected, AvatarStore  # noqa: E402


def _print_record(record, report=None) -> None:
    print(f"  avatarId : {record.avatar_id}")
    print(f"  file     : {record.path}")
    print(f"  size     : {record.width}x{record.height}")
    print(f"  usable   : {record.usable} ({record.usability_reason})")
    if report is not None and report.analysis is not None:
        pose = report.analysis.head_pose
        print(
            f"  face     : {report.analysis.landmark_count} landmarks, "
            f"yaw {pose.yaw:+.1f} pitch {pose.pitch:+.1f} roll {pose.roll:+.1f}"
        )
        for issue in report.warnings:
            print(f"  warning  : {issue.message}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--synthetic", action="store_true", help="generate a face with Stable Diffusion 1.5")
    action.add_argument("--human", metavar="FILE", help="register a photo of a real person")
    action.add_argument("--list", action="store_true", help="list the avatar store")
    action.add_argument("--delete", metavar="ID", help="delete an avatar")

    parser.add_argument("--avatar-id", help="id used as AvatarRenderJob.avatarId")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--notes", default="")
    # human
    parser.add_argument("--subject", default="", help="who is in the photo")
    parser.add_argument("--consent", default="", help=f"one of {sorted(provenance.FACE_CONSENT_BASES)}")
    parser.add_argument("--licence", default="")
    # synthetic
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=6, help="seeds to try until one passes the quality gate")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args(argv)

    store = AvatarStore()

    if args.list:
        records = store.list()
        if not records:
            print(f"No avatars in {store.root}. Create one with --synthetic or --human.")
            return 0
        for record in records:
            mark = "usable " if record.usable else "BLOCKED"
            print(f"  {mark}  {record.avatar_id:<24} {record.width}x{record.height}  {record.usability_reason}")
        return 0

    if args.delete:
        try:
            store.delete(args.delete)
        except (AvatarError, AvatarNotFound) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(f"Deleted avatar {args.delete}")
        return 0

    if not args.avatar_id:
        parser.error("--avatar-id is required")

    try:
        if args.human:
            record, report = store.register(
                args.human,
                avatar_id=args.avatar_id,
                source=provenance.HUMAN,
                subject=args.subject,
                licence=args.licence,
                consent_basis=args.consent,
                notes=args.notes,
                overwrite=args.overwrite,
            )
        else:
            import avatar_generator
            from face_engine import FACE_ENGINE_LOCK, shared_face_engine

            def check(image):
                with FACE_ENGINE_LOCK:
                    return shared_face_engine().check_quality(image)

            print(f"Generating a synthetic face (seed {args.seed}, up to {args.attempts} attempts)...")
            generated = avatar_generator.generate_registered_avatar(
                store,
                check,
                args.avatar_id,
                prompt=args.prompt or avatar_generator.DEFAULT_PROMPT,
                seed=args.seed,
                attempts=args.attempts,
                steps=args.steps,
                overwrite=args.overwrite,
            )
            for seed, reason in generated.rejected_seeds.items():
                print(f"  seed {seed} rejected: {reason}")
            print(f"  accepted seed {generated.seed}")
            record, report = store.get(args.avatar_id), None
    except AvatarRejected as err:
        print("Photo rejected by the quality gate:", file=sys.stderr)
        for issue in err.report.errors:
            print(f"  - {issue.message}", file=sys.stderr)
        return 2
    except (AvatarError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print("Registered avatar:")
    _print_record(record, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
