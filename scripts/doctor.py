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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

EXPECTED_PYTHON = (3, 10)
PROJECT_ENV = PROJECT_ROOT / "backend" / ".conda"
MIN_FREE_GB = 5.0


@dataclass
class Result:
    area: str
    check: str
    status: str  # PASS | WARN | FAIL
    detail: str
    fix: str = ""


results: List[Result] = []


def record(area: str, check: str, status: str, detail: str, fix: str = "") -> None:
    results.append(Result(area, check, status, detail, fix))


def guarded(area: str, check: str, fn: Callable[[], None]) -> None:
    """Run one check; an unexpected error becomes a FAIL, never a crash."""
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        record(area, check, "FAIL", f"{type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_interpreter() -> None:
    version = sys.version_info[:2]
    in_project_env = Path(sys.prefix).resolve() == PROJECT_ENV.resolve()
    detail = f"Python {sys.version.split()[0]} at {sys.prefix}"
    if version != EXPECTED_PYTHON:
        record(
            "environment", "interpreter", "FAIL", detail,
            "Use ./backend/.conda/bin/python (Python 3.10); base conda cannot run this project",
        )
    elif not in_project_env:
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
    ("cv2", True),
    ("speechbrain", False),
    ("uroman", False),
    ("TTS", False),
]


def check_packages() -> None:
    for name, required in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            status, fix = "PASS", ""
            if name == "numpy" and int(str(version).split(".")[0]) >= 2:
                status, fix = "FAIL", 'numpy 2 breaks Coqui TTS: pip install "numpy<2.0.0"'
            record("packages", name, status, f"version {version}", fix)
        except Exception as err:  # noqa: BLE001
            record(
                "packages", name, "FAIL" if required else "WARN",
                f"not importable ({type(err).__name__})",
                "pip install -r backend/requirements.txt",
            )


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        record("hardware", "cuda", "WARN", "CUDA not available; everything runs on CPU",
               "Check the NVIDIA driver and the CUDA build of torch")
        return
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    detail = f"{name}, {free / 2**20:.0f} of {total / 2**20:.0f} MiB free"
    status = "PASS" if free > 3 * 2**30 else "WARN"
    record("hardware", "cuda", status, detail,
           "" if status == "PASS" else "Another process holds VRAM; close it before rendering")


def check_disk() -> None:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    status = "PASS" if free_gb >= 15 else ("WARN" if free_gb >= MIN_FREE_GB else "FAIL")
    record("hardware", "disk", status, f"{free_gb:.1f} GB free",
           "" if status == "PASS" else "Free space before fetching models (full set ≈ 15 GB)")


def check_speech_weights() -> None:
    from model_registry import audit_model_weights

    for status in audit_model_weights():
        # Kokoro is the one model everything else falls back to.
        severity = "FAIL" if status.key == "kokoro" else "WARN"
        record(
            "models", status.key,
            "PASS" if status.present else severity,
            f"{status.size_label} — {status.detail}",
            "" if status.present else f"scripts/fetch_models.py --only {status.key}",
        )


def check_other_weights() -> None:
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

    ecapa = PROJECT_ROOT / ".models" / "ecapa" / "embedding_model.ckpt"
    record("models", "ecapa", "PASS" if ecapa.is_file() else "WARN",
           "speaker-similarity encoder" if ecapa.is_file() else "missing",
           "" if ecapa.is_file() else "Loaded on first similarity call; needs network once")


def check_references() -> None:
    import provenance

    audio_ext = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    image_ext = {".jpg", ".jpeg", ".png", ".webp"}
    inputs = PROJECT_ROOT / "inputs"

    def report(kind: str, files: List[Path], how_to_add: str) -> None:
        if not files:
            record("inputs", kind, "WARN", "none present", how_to_add)
            return
        admissible = 0
        for path in files:
            info = provenance.describe(path)
            admissible += bool(info["admissible"])
        status = "PASS" if admissible else "WARN"
        record("inputs", kind, status,
               f"{len(files)} file(s), {admissible} admissible as evidence",
               "" if admissible else "Add a human reference with a consent basis (provenance sidecar)")

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
        usable = sum(provenance.usability(p)[0] for p in faces)
        record("inputs", "avatar faces", "PASS" if usable else "WARN",
               f"{len(faces)} image(s), {usable} usable (provenance permits animation)",
               "" if usable == len(faces) else
               "Register unrecorded images with scripts/make_avatar.py, or remove them")


def check_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        record("hardware", binary, "PASS" if path else "FAIL",
               path or "not on PATH; videos cannot be encoded",
               "" if path else "sudo apt install ffmpeg")


def check_config() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.is_file():
        record("config", ".env", "FAIL", "missing", "cp .env.example .env")
        return
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    queue = values.get("QUEUE_BACKEND", "in_memory")
    auth = values.get("AUTH_ENABLED", "false").lower() == "true"
    detail = f"QUEUE_BACKEND={queue}, AUTH_ENABLED={auth}"
    if auth and not values.get("API_KEYS"):
        record("config", ".env", "FAIL", detail, "AUTH_ENABLED with empty API_KEYS rejects every request")
    else:
        record("config", ".env", "PASS", detail)
    if queue == "celery":
        check_redis(values)


def check_redis(values: dict) -> None:
    import socket

    host = values.get("REDIS_HOST", "localhost")
    port = int(values.get("REDIS_PORT", "6379"))
    try:
        with socket.create_connection((host, port), timeout=1):
            record("config", "redis", "PASS", f"{host}:{port} reachable")
    except OSError:
        record("config", "redis", "FAIL", f"{host}:{port} unreachable",
               "./start-docker.sh, or set QUEUE_BACKEND=in_memory")


def check_api() -> None:
    url = os.getenv("AVATAR_API", "http://localhost:8000") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read())
        weights = payload.get("modelWeights", {})
        record("services", "api", "PASS",
               f"{url} ok, {weights.get('available', '?')}/{weights.get('total', '?')} models")
    except Exception:  # noqa: BLE001
        record("services", "api", "WARN", f"{url} not responding",
               "cd backend && PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000")


def check_frontend() -> None:
    modules = PROJECT_ROOT / "frontend" / "node_modules"
    record("services", "frontend deps", "PASS" if modules.is_dir() else "WARN",
           "node_modules present" if modules.is_dir() else "not installed",
           "" if modules.is_dir() else "cd frontend && npm install")


def check_docs() -> None:
    pdf = PROJECT_ROOT / "docs" / "4895e15d-8adb-4146-a985-52babca3b3c5_AI_Avatar_Creation_Platform_using_Open_Source_Tech.pdf"
    if pdf.is_file() and b"\xef\xbf\xbd" * 4 in pdf.read_bytes():
        record("docs", "problem statement", "WARN", "PDF is corrupted (bytes replaced by U+FFFD)",
               "Get an intact copy from the original source")
    elif pdf.is_file():
        record("docs", "problem statement", "PASS", "PDF present")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    for area, check, fn in CHECKS:
        guarded(area, check, fn)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        marks = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}
        current = None
        for r in results:
            if r.area != current:
                current = r.area
                print(f"\n{current.upper()}")
            print(f"  {marks[r.status]:<5} {r.check:<18} {r.detail}")
            if r.fix:
                print(f"        fix: {r.fix}")
        counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL")}
        print(f"\n{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
