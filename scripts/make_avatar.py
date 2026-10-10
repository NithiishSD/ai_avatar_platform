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

What it writes: the image and a provenance sidecar (who is in it, on what basis it may be
used, under what licence) in the avatar store, ``inputs/faces/``. The ``--avatar-id`` is the
name later passed as ``AvatarRenderJob.avatarId``.

Exit codes: 0 on success, 1 on a store or generation error, 2 when the quality gate rejected
the photo. A distinct code for "rejected" lets a calling script tell "bad photo" from "broken
setup" without parsing the message.

Concepts used here:

**A provenance record** is the consent paper trail for a face. Golden rule 3 says no face is
used without one, so registration and the record are one step that cannot be split.

**A mutually exclusive group** in argparse is a set of flags of which at most one may be given;
``required=True`` makes exactly one mandatory. argparse prints the error itself, so the script
needs no "did you pass two actions?" checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# The repository root: scripts/ is one level below it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Put backend/ on the import path so its modules import by name, the way the server imports them.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# These imports must come after the sys.path change above, which is why ruff's
# "imports at top of file" rule (E402) is silenced on them.
import provenance  # noqa: E402
from avatar_store import AvatarError, AvatarNotFound, AvatarRejected, AvatarStore  # noqa: E402


def _print_record(record, report=None) -> None:
    """
    Print one stored avatar, plus the face analysis when a quality report is available.

    ``report`` is None for a synthetic face (the generator ran the gate itself) and for
    anything not just checked, so the face lines are optional.
    """
    print(f"  avatarId : {record.avatar_id}")
    print(f"  file     : {record.path}")
    print(f"  size     : {record.width}x{record.height}")
    print(f"  usable   : {record.usable} ({record.usability_reason})")
    if report is not None and report.analysis is not None:
        # Head pose in degrees: yaw turns left/right, pitch nods up/down, roll tilts sideways.
        # The :+.1f format always shows the sign, so a turn to either side is readable.
        pose = report.analysis.head_pose
        print(
            f"  face     : {report.analysis.landmark_count} landmarks, "
            f"yaw {pose.yaw:+.1f} pitch {pose.pitch:+.1f} roll {pose.roll:+.1f}"
        )
        # Warnings did not block registration (errors would have); they are shown so a
        # borderline photo can be swapped for a better one.
        for issue in report.warnings:
            print(f"  warning  : {issue.message}")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run one store command (list, delete, register a human photo or generate a synthetic face).

    ``argv`` defaults to the real command line; a test can pass a list instead. Returns the exit code.
    """
    # description=__doc__ reuses the module docstring as --help; the Raw formatter keeps its line breaks.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Exactly one action per run: the four commands do unrelated things.
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--synthetic", action="store_true", help="generate a face with Stable Diffusion 1.5")
    action.add_argument("--human", metavar="FILE", help="register a photo of a real person")
    action.add_argument("--list", action="store_true", help="list the avatar store")
    action.add_argument("--delete", metavar="ID", help="delete an avatar")

    # Options shared by both registration paths. --avatar-id is not marked required because
    # --list and --delete do not need it; it is checked by hand further down.
    parser.add_argument("--avatar-id", help="id used as AvatarRenderJob.avatarId")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--notes", default="")
    # human: the consent fields. Empty defaults let the store, not argparse, refuse a missing
    # value, so the refusal message comes from the one place that owns the consent rule.
    parser.add_argument("--subject", default="", help="who is in the photo")
    parser.add_argument("--consent", default="", help=f"one of {sorted(provenance.FACE_CONSENT_BASES)}")
    parser.add_argument("--licence", default="")
    # synthetic: Stable Diffusion settings. A seed makes a generated face reproducible.
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=6, help="seeds to try until one passes the quality gate")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args(argv)

    # The store owns inputs/faces/ and its sidecars; every command below goes through it.
    store = AvatarStore()

    if args.list:
        records = store.list()
        if not records:
            print(f"No avatars in {store.root}. Create one with --synthetic or --human.")
            return 0
        for record in records:
            # The trailing space in "usable " pads it to the width of "BLOCKED", so columns line up.
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

    # Both registration paths need an id. parser.error prints usage and exits with code 2.
    if not args.avatar_id:
        parser.error("--avatar-id is required")

    try:
        if args.human:
            # register() runs the quality gate and validates the consent fields before writing,
            # so nothing lands on disk for a refused photo.
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
            # Imported only on this path: it pulls in Stable Diffusion, which is slow to import
            # and unneeded for --human, --list and --delete.
            import avatar_generator
            from face_engine import FACE_ENGINE_LOCK, shared_face_engine

            def check(image):
                """Run the face quality gate on one generated image, under the engine's lock."""
                # The face landmarker is one shared native object not documented as thread-safe,
                # so every use holds FACE_ENGINE_LOCK (see backend/face_engine.py).
                with FACE_ENGINE_LOCK:
                    return shared_face_engine().check_quality(image)

            # The generator tries --attempts seeds, starting at --seed, until one passes `check`.
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
            # The generator registered the face itself; read the stored record back. There is no
            # report to show because the gate ran inside the generator.
            record, report = store.get(args.avatar_id), None
    # Order matters: AvatarRejected is caught first so it gets its own message and exit code
    # before the broader handler below can claim it.
    except AvatarRejected as err:
        print("Photo rejected by the quality gate:", file=sys.stderr)
        for issue in err.report.errors:
            print(f"  - {issue.message}", file=sys.stderr)
        return 2
    # RuntimeError covers a generation failure, such as a missing model.
    except (AvatarError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print("Registered avatar:")
    _print_record(record, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
