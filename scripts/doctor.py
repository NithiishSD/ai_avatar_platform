#!/usr/bin/env python
"""
One-command health check for the whole project.

    PYTHONPATH=backend backend/.conda/bin/python scripts/doctor.py [--json]

Each check prints PASS, WARN or FAIL with the fix for anything not passing.
Exit code is 1 when any check FAILs, so it also works as a CI gate.

It never downloads anything and never loads a model onto the GPU: it reads
what is on disk, what is importable, and whether the API answers. Checks are
independent, so one broken area never hides the state of the others -- the
same lesson as the model weight audit.

What it writes: nothing to disk. It prints a grouped report to stdout, or a
JSON list of results with ``--json`` (one object per check, with the fields of
``Result`` below), so another tool can parse it.

The three severities mean:
  * PASS - this part is ready.
  * WARN - the project still runs, but something is degraded or optional
    (no GPU, an optional package, no reference files yet).
  * FAIL - something the project cannot work without. Any FAIL makes the
    exit code 1; the project's Definition of Done requires 0 FAIL.

How the script is built: every check is a small function that calls
``record()`` one or more times. ``main()`` runs them all through
``guarded()``, collects the results in a module-level list, then prints them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

# parents[1] of scripts/doctor.py is the repository root. resolve() first turns
# a relative or symlinked path into an absolute one, so this works from any
# working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Putting backend/ at the front of the import path lets the checks import the
# project's own modules (model_registry, provenance) even when PYTHONPATH was
# not set on the command line.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# The project pins Python 3.10 (see the project constraints); base conda
# is 3.14 and cannot install the pinned packages.
EXPECTED_PYTHON = (3, 10)
# The project's own conda environment. Comparing sys.prefix with it tells us
# whether this script runs inside that environment.
PROJECT_ENV = PROJECT_ROOT / "backend" / ".conda"
# Below this many GB of free disk the disk check FAILs instead of warning.
MIN_FREE_GB = 5.0


@dataclass
class Result:
    """
    One line of the report.

    A ``dataclass`` generates ``__init__`` and ``__repr__`` from the annotated
    fields, so ``Result(area, check, status, detail)`` just works. It also lets
    ``dataclasses.asdict`` turn a result into a plain dict for ``--json``.
    """

    # Grouping heading in the printed report, such as "models" or "hardware".
    area: str
    # The specific thing checked inside that area, such as "cuda".
    check: str
    status: str  # PASS | WARN | FAIL
    # What was found, in words a person can read.
    detail: str
    # How to fix it. Empty for a PASS; golden rule 7 says errors explain
    # their own fix, so every WARN or FAIL should fill this in.
    fix: str = ""


# Module-level list that every check appends to. A global keeps each check a
# zero-argument function, which is all ``guarded()`` needs to call.
results: List[Result] = []


def record(area: str, check: str, status: str, detail: str, fix: str = "") -> None:
    """Append one result to the report."""
    results.append(Result(area, check, status, detail, fix))


def guarded(area: str, check: str, fn: Callable[[], None]) -> None:
    """Run one check; an unexpected error becomes a FAIL, never a crash."""
    try:
        fn()
    # Catching every Exception is deliberate here (hence the noqa for the
    # "blind except" lint rule): a bug or missing import inside one check must
    # be reported as that check failing, not stop every check after it.
    except Exception as err:  # noqa: BLE001
        record(area, check, "FAIL", f"{type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_interpreter() -> None:
    """FAIL on the wrong Python version; WARN on 3.10 outside backend/.conda."""
    # version_info is a tuple like (3, 10, 14, 'final', 0); the first two
    # items are major and minor, which is all the pin cares about.
    version = sys.version_info[:2]
    # sys.prefix is the root of the running environment. Both sides are
    # resolved so a symlinked path still compares equal.
    in_project_env = Path(sys.prefix).resolve() == PROJECT_ENV.resolve()
    detail = f"Python {sys.version.split()[0]} at {sys.prefix}"
    if version != EXPECTED_PYTHON:
        record(
            "environment", "interpreter", "FAIL", detail,
            "Use ./backend/.conda/bin/python (Python 3.10); base conda cannot run this project",
        )
    elif not in_project_env:
        # Right version, different environment: it may work, but installed
        # package versions can differ from the pinned ones.
        record(
            "environment", "interpreter", "WARN", detail,
            "Python 3.10 but not backend/.conda; packages may differ",
        )
    else:
        record("environment", "interpreter", "PASS", detail)


PACKAGES = [
    # (import name, required: missing is FAIL rather than WARN)
    ("torch", True),
    ("torchaudio", True),
    ("numpy", True),
    ("fastapi", True),
    ("pydantic", True),
    ("celery", True),
    ("transformers", True),
    ("soundfile", True),
    ("kokoro", True),
    ("mediapipe", True),
    # OpenCV's import name is cv2, not opencv.
    ("cv2", True),
    # Optional: these only power specific features (speaker similarity,
    # romanising non-Latin text, Coqui TTS / XTTS), so missing is a WARN.
    ("speechbrain", False),
    ("uroman", False),
    ("TTS", False),
]


def check_packages() -> None:
    """Import each package in PACKAGES and report its version."""
    for name, required in PACKAGES:
        try:
            # import_module takes the module name as a string, so the list
            # above can drive the imports instead of one import line each.
            module = importlib.import_module(name)
            # Not every package defines __version__; "?" keeps the report
            # readable instead of raising.
            version = getattr(module, "__version__", "?")
            status, fix = "PASS", ""
            # Importable is not enough for numpy: the project pins numpy<2
            # because Coqui TTS and numba break on numpy 2.
            if name == "numpy" and int(str(version).split(".")[0]) >= 2:
                status, fix = "FAIL", 'numpy 2 breaks Coqui TTS: pip install "numpy<2.0.0"'
            record("packages", name, status, f"version {version}", fix)
        # Any exception, not just ImportError: a broken install can fail with
        # other errors while its own imports run.
        except Exception as err:  # noqa: BLE001
            record(
                "packages", name, "FAIL" if required else "WARN",
                f"not importable ({type(err).__name__})",
                "pip install -r backend/requirements.txt",
            )


def check_gpu() -> None:
    """Report whether CUDA is usable and how much VRAM is free right now."""
    # Imported inside the function: if torch is missing, guarded() turns the
    # ImportError into a FAIL for this check only.
    import torch

    if not torch.cuda.is_available():
        # A WARN, not a FAIL: every model also runs on CPU, only slower.
        record("hardware", "cuda", "WARN", "CUDA not available; everything runs on CPU",
               "Check the NVIDIA driver and the CUDA build of torch")
        return
    # mem_get_info() returns (free, total) in bytes for the current device. It
    # asks the driver, so it also sees memory held by other processes.
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    # 2**20 bytes is one MiB, the unit nvidia-smi uses.
    detail = f"{name}, {free / 2**20:.0f} of {total / 2**20:.0f} MiB free"
    # 2**30 bytes is one GiB. Less than 3 GiB free on a 6 GB card means
    # another process is holding VRAM a render would need.
    status = "PASS" if free > 3 * 2**30 else "WARN"
    record("hardware", "cuda", status, detail,
           "" if status == "PASS" else "Another process holds VRAM; close it before rendering")


def check_disk() -> None:
    """PASS at 15 GB free (room for the full model set), FAIL under MIN_FREE_GB."""
    # disk_usage reports the filesystem that holds the project. 1e9 gives
    # decimal GB, matching the "≈ 15 GB" in the fix text.
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    status = "PASS" if free_gb >= 15 else ("WARN" if free_gb >= MIN_FREE_GB else "FAIL")
    record("hardware", "disk", status, f"{free_gb:.1f} GB free",
           "" if status == "PASS" else "Free space before fetching models (full set ≈ 15 GB)")


def check_speech_weights() -> None:
    """Report each speech model's weights using the registry's own audit."""
    # Reusing the registry's audit means the doctor and /health can never
    # disagree about which weights are present.
    from model_registry import audit_model_weights

    for status in audit_model_weights():
        # Kokoro is the one model everything else falls back to.
        severity = "FAIL" if status.key == "kokoro" else "WARN"
        record(
            "models", status.key,
            "PASS" if status.present else severity,
            f"{status.size_label} — {status.detail}",
            # The registry knows the right fix: a fetch, or "none on this stack".
            "" if status.present else status.fix,
        )


def check_other_weights() -> None:
    """Report vision weights and the ECAPA speaker-similarity encoder."""
    from model_registry import audit_vision_weights

    for status in audit_vision_weights():
        # Without the landmarker no face can be analysed or rendered at all.
        severity = "FAIL" if status.key == "face-landmarker" else "WARN"
        record(
            "models", status.key,
            "PASS" if status.present else severity,
            f"{status.size_label} — {status.detail}",
            "" if status.present else f"scripts/fetch_vision_models.py --only {status.key}",
        )

    # ECAPA-TDNN is the speaker-embedding model behind voice similarity scores.
    # It is not in either registry audit, so it is checked by its file path.
    ecapa = PROJECT_ROOT / ".models" / "ecapa" / "embedding_model.ckpt"
    record("models", "ecapa", "PASS" if ecapa.is_file() else "WARN",
           "speaker-similarity encoder" if ecapa.is_file() else "missing",
           "" if ecapa.is_file() else "Loaded on first similarity call; needs network once")


def check_references() -> None:
    """
    Count voice and face inputs and how many carry a usable provenance record.

    Golden rule 3 (consent before use): a file only counts if its provenance
    sidecar says it may be used. The ``provenance`` module reads that sidecar.
    """
    import provenance

    # Sets for O(1) membership tests; suffixes are lower-cased before the test
    # so "PHOTO.JPG" matches too.
    audio_ext = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    image_ext = {".jpg", ".jpeg", ".png", ".webp"}
    inputs = PROJECT_ROOT / "inputs"

    def report(kind: str, files: List[Path], how_to_add: str) -> None:
        """Record one result for a group of reference files."""
        if not files:
            record("inputs", kind, "WARN", "none present", how_to_add)
            return
        admissible = 0
        for path in files:
            info = provenance.describe(path)
            # bool is a subclass of int in Python, so True adds 1.
            admissible += bool(info["admissible"])
        # "Admissible" means usable as evidence for a measurement (a consented
        # human reference); a synthetic or smoke-test file is not.
        status = "PASS" if admissible else "WARN"
        record("inputs", kind, status,
               f"{len(files)} file(s), {admissible} admissible as evidence",
               "" if admissible else "Add a human reference with a consent basis (provenance sidecar)")

    # Voice references sit directly in inputs/; faces sit in inputs/faces/.
    voices = [p for p in inputs.glob("*") if p.suffix.lower() in audio_ext]
    faces = [p for p in (inputs / "faces").glob("*") if p.suffix.lower() in image_ext]
    report("voice references", voices,
           "scripts/make_reference.py --human FILE ... (or --smoke for pipeline tests)")
    # For a face the question is "may it be animated", not "is it evidence":
    # a synthetic face is usable for rendering even though it proves nothing.
    if not faces:
        record("inputs", "avatar faces", "WARN", "none present",
               "scripts/make_avatar.py --synthetic --avatar-id demo  (or --human FILE ...)")
    else:
        # usability() returns a tuple whose first item is the yes/no answer;
        # summing those booleans counts the usable images.
        usable = sum(provenance.usability(p)[0] for p in faces)
        record("inputs", "avatar faces", "PASS" if usable else "WARN",
               f"{len(faces)} image(s), {usable} usable (provenance permits animation)",
               "" if usable == len(faces) else
               "Register unrecorded images with scripts/make_avatar.py, or remove them")


def check_ffmpeg() -> None:
    """ffmpeg encodes the output videos; ffprobe reads media durations."""
    for binary in ("ffmpeg", "ffprobe"):
        # shutil.which searches PATH like the shell does and returns the full
        # path, or None when the program is not installed.
        path = shutil.which(binary)
        record("hardware", binary, "PASS" if path else "FAIL",
               path or "not on PATH; videos cannot be encoded",
               "" if path else "sudo apt install ffmpeg")


def check_config() -> None:
    """Read .env by hand and flag settings that would break the API."""
    env = PROJECT_ROOT / ".env"
    if not env.is_file():
        record("config", ".env", "FAIL", "missing", "cp .env.example .env")
        return
    # A tiny KEY=VALUE parser instead of python-dotenv, so this check works
    # even when that package is not installed. Comment lines are skipped.
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            # partition splits on the first "=" only, so a value that itself
            # contains "=" (a base64 key, say) survives intact.
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    # The defaults mirror the server's own defaults for an unset variable.
    queue = values.get("QUEUE_BACKEND", "in_memory")
    auth = values.get("AUTH_ENABLED", "false").lower() == "true"
    detail = f"QUEUE_BACKEND={queue}, AUTH_ENABLED={auth}"
    if auth and not values.get("API_KEYS"):
        record("config", ".env", "FAIL", detail, "AUTH_ENABLED with empty API_KEYS rejects every request")
    else:
        record("config", ".env", "PASS", detail)
    # Redis only matters when Celery is the queue; the in-memory queue does
    # not use it, so it is not checked in that case.
    if queue == "celery":
        check_redis(values)


def check_redis(values: dict) -> None:
    """Open a TCP connection to Redis to prove it is listening."""
    import socket

    host = values.get("REDIS_HOST", "localhost")
    port = int(values.get("REDIS_PORT", "6379"))
    try:
        # A plain TCP connect is enough to prove something listens on the
        # port, without needing the redis client package. The one-second
        # timeout keeps the doctor fast when the host is unreachable.
        with socket.create_connection((host, port), timeout=1):
            record("config", "redis", "PASS", f"{host}:{port} reachable")
    except OSError:
        record("config", "redis", "FAIL", f"{host}:{port} unreachable",
               "./start-docker.sh, or set QUEUE_BACKEND=in_memory")


def check_api() -> None:
    """Call the running API's /health endpoint, if a server is up."""
    # AVATAR_API lets the doctor check a server on another host or port.
    url = os.getenv("AVATAR_API", "http://localhost:8000") + "/health"
    try:
        # urllib from the standard library, so no HTTP client package is
        # needed. A short timeout: a server that is not running should not
        # make the doctor hang.
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read())
        weights = payload.get("modelWeights", {})
        record("services", "api", "PASS",
               f"{url} ok, {weights.get('available', '?')}/{weights.get('total', '?')} models")
    # A WARN rather than a FAIL: the server being down is a normal state when
    # you are not using it, and the doctor should still pass.
    except Exception:  # noqa: BLE001
        record("services", "api", "WARN", f"{url} not responding",
               "cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000")


def check_frontend() -> None:
    """Report whether the frontend's npm dependencies are installed."""
    modules = PROJECT_ROOT / "frontend" / "node_modules"
    record("services", "frontend deps", "PASS" if modules.is_dir() else "WARN",
           "node_modules present" if modules.is_dir() else "not installed",
           "" if modules.is_dir() else "cd frontend && npm install")


def check_docs() -> None:
    """Warn when the problem-statement PDF has been mangled."""
    pdf = PROJECT_ROOT / "docs" / "4895e15d-8adb-4146-a985-52babca3b3c5_AI_Avatar_Creation_Platform_using_Open_Source_Tech.pdf"
    # b"\xef\xbf\xbd" is U+FFFD (the replacement character) encoded as UTF-8.
    # A binary file that was once decoded and re-encoded as text has its
    # invalid bytes replaced by it, so four in a row is a sign of corruption.
    if pdf.is_file() and b"\xef\xbf\xbd" * 4 in pdf.read_bytes():
        record("docs", "problem statement", "WARN", "PDF is corrupted (bytes replaced by U+FFFD)",
               "Get an intact copy from the original source")
    elif pdf.is_file():
        record("docs", "problem statement", "PASS", "PDF present")
    # No else: an absent PDF records nothing, because docs/ is not shipped on
    # every branch.


# The run order, which is also the order of the printed report. Each entry is
# (area, check, function); the area and check name are only used to label a
# FAIL when the function itself crashes.
CHECKS = [
    ("environment", "interpreter", check_interpreter),
    ("packages", "imports", check_packages),
    ("hardware", "cuda", check_gpu),
    ("hardware", "disk", check_disk),
    ("hardware", "ffmpeg", check_ffmpeg),
    ("models", "speech weights", check_speech_weights),
    ("models", "other weights", check_other_weights),
    ("inputs", "references", check_references),
    ("config", ".env", check_config),
    ("services", "api", check_api),
    ("services", "frontend", check_frontend),
    ("docs", "problem statement", check_docs),
]


def main() -> int:
    """Run every check, print the report, and return the exit code."""
    # argparse builds the command line: it parses sys.argv, generates --help
    # from the description and help strings, and rejects unknown flags.
    # store_true makes --json a flag that is False unless given.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    for area, check, fn in CHECKS:
        guarded(area, check, fn)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        marks = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}
        # Print an area heading whenever the area changes. Results arrive
        # grouped because CHECKS is ordered by area.
        current = None
        for r in results:
            if r.area != current:
                current = r.area
                print(f"\n{current.upper()}")
            # :<5 and :<18 left-align in fixed-width columns so the details
            # line up.
            print(f"  {marks[r.status]:<5} {r.check:<18} {r.detail}")
            if r.fix:
                print(f"        fix: {r.fix}")
        # Summing booleans counts the True values, one count per status.
        counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL")}
        print(f"\n{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")

    # Only FAIL changes the exit code; warnings never break a CI gate.
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    # SystemExit with an int sets the process exit code, which is what a
    # shell or CI job reads.
    raise SystemExit(main())
