"""
SadTalker: a talking head with real head motion, from one photo and the audio (I-01, option 2).

The third render engine. The blendshape warp and Wav2Lip move a still photo in 2-D: the lips, a
little jaw, blinks and the small head sway of ``head_motion``. SadTalker (Zhang et al., CVPR 2023,
"Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face
Animation") instead predicts how a 3-D face would move while saying the audio: head pose, jaw,
cheeks and expression together. That is what the owner asked for after watching the Wav2Lip video
(only the lips moved).

Concepts used here, explained once:

**3DMM coefficients.** A 3-D Morphable Model describes a face as an average 3-D face plus weighted
sums of a few dozen "shape", "expression" and "pose" directions. SadTalker never draws a 3-D mesh:
it fits the photo's coefficients once, predicts how the expression and pose coefficients change over
the audio (ExpNet for the mouth and cheeks, PoseVAE for the head), and then a face renderer
(face-vid2vid) warps the photo to match each frame's coefficients.

**Pose style.** PoseVAE is a generator conditioned on one of 46 learned "styles" of head movement
(some nod more, some sway). We pick one per job from the job id, so a re-render is the same video and
different jobs do not all move alike.

**Whole-picture animation (``resize`` mode).** SadTalker can either animate a crop around the face
and paste it back into the still photo (``full``), or animate the whole picture shrunk to 256x256
(``resize``). The first version used ``full``; the owner watched it (M-12) and saw the shoulders
and the outer hair stand still while the head moved, with the crop's edge out of step with the
face. In ``resize`` mode the face renderer moves everything it sees together (head, hair, neck,
shoulders), as in the head-and-shoulders videos it was trained on, so there is no paste-back edge.
Measured on ``demo``: movement in the bottom (shoulder) band 24.8 grey levels vs 20.4 with ``full``,
background corners 0.9 vs 2.2.

**Square padding.** ``resize`` squeezes any picture into a square, which would distort a portrait or
landscape photo's face. So the photo is first padded to a square with its own edge pixels repeated
outward (repeated, not mirrored: a mirrored border could contain a mirrored half-face for the
detector to find), and the caller cuts each frame back to the photo's area (``SadTalkerResult.box``).
A photo whose face is small in a large frame comes out soft, because the whole square is animated at
256 px.

**Sharpening.** The 256 px frames are enlarged by Real-ESRGAN in the render engine
(``render_engine._sadtalker_frames``), which restores most of the photo's sharpness (Laplacian
variance 91 vs the photo's 103 and plain resizing's 18 on ``demo``) at ~36 ms a frame on the GPU.

**Why a separate process.** SadTalker is run as ``python sadtalker_engine.py --child ...`` instead of
being imported into the API server, for three reasons:

* its GPU memory (about 2-3 GB) is returned to the card when the process exits, which is the
  6 GB rule (golden rule 5) without having to trust its code to release anything;
* its 2023 code needs four small compatibility patches (below) that must not leak into the server;
* it reads its config and weights from relative paths, so it must run with its own folder as the
  working directory.

The cost is load time: the models are loaded for every render (a few seconds), which is small next
to the render itself.

There is no fallback in here (golden rule 1). Missing code, weights or packages raise with the fetch
command; the engine name in the result is ``sadtalker`` only when SadTalker made the frames.

Licence: SadTalker's code and weights are Apache-2.0; the two face-detection weights come from
facexlib (MIT). None of them is licence-gated, so they are fetched like any default-off model.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The upstream checkout lives in .models/ like every other model, laid out exactly as upstream expects
# (checkpoints/ and gfpgan/weights/ inside it), because SadTalker opens those paths relative to its
# working directory.
SADTALKER_DIR = Path(__file__).resolve().parents[1] / ".models" / "sadtalker"
SADTALKER_REPO = "https://github.com/OpenTalker/SadTalker.git"
# Pinned to one commit (the last on main, 11 Oct 2023): the code is executed, so only reviewed code runs.
SADTALKER_COMMIT = "cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8"

_RELEASE = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
_FACEXLIB = "https://github.com/xinntao/facexlib/releases/download/v0.1.0"
# path inside SADTALKER_DIR -> (URL, SHA-256, exact size in bytes). Every file is pinned by hash: the
# .pth files are opened with torch.load, which can run code from a tampered pickle (D-56).
# Only the 256 px model: the 512 px one doubles the time and the memory for a crop that is pasted
# back into a frame of at most 1080 px anyway.
SADTALKER_FILES: Dict[str, Tuple[str, str, int]] = {
    "checkpoints/SadTalker_V0.0.2_256.safetensors": (
        f"{_RELEASE}/SadTalker_V0.0.2_256.safetensors",
        "c211f5d6de003516bf1bbda9f47049a4c9c99133b1ab565c6961e5af16477bff", 725_066_984),
    # The mapping net turns coefficients into the renderer's motion; ``resize`` mode uses 00229
    # (00109 belongs to ``full`` mode, which is not used).
    "checkpoints/mapping_00229-model.pth.tar": (
        f"{_RELEASE}/mapping_00229-model.pth.tar",
        "62a1e06006cc963220f6477438518ed86e9788226c62ae382ddc42fbcefb83f1", 155_521_183),
    # Face detector (RetinaFace) and 98-point landmark model used to crop and align the photo.
    "gfpgan/weights/detection_Resnet50_Final.pth": (
        f"{_FACEXLIB}/detection_Resnet50_Final.pth",
        "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d", 109_497_761),
    "gfpgan/weights/alignment_WFLW_4HG.pth": (
        f"{_FACEXLIB}/alignment_WFLW_4HG.pth",
        "bbfd137307a4c7debd5c283b9b0ce539466cee417ac0a155e184d857f9f2899c", 193_670_248),
}

# Python module name -> pip requirement. facexlib and filterpy go in with --no-deps: their declared
# dependency opencv-python would install a third copy of OpenCV over the two already here.
SADTALKER_PACKAGES = {
    "kornia": "kornia==0.7.4",
    "yacs": "yacs==0.1.8",
    "pydub": "pydub==0.25.1",
    "imageio_ffmpeg": "imageio-ffmpeg==0.5.1",
    "facexlib": "facexlib==0.3.0",
    "filterpy": "filterpy==1.4.5",
}
SADTALKER_PIP = (
    "pip install kornia==0.7.4 yacs==0.1.8 pydub==0.25.1 imageio-ffmpeg==0.5.1 "
    "&& pip install --no-deps facexlib==0.3.0 filterpy==1.4.5"
)
FETCH_COMMAND = "python scripts/fetch_vision_models.py --only sadtalker"

# Freed on the card (by unloading this process's other models) before the child starts.
# Measured: see docs/12-PROGRESS.md, I-01 SadTalker entry.
REQUIRED_VRAM_MB = 3000
# A render that takes longer than this is treated as hung. Generous: CPU renders run ~10x real time.
TIMEOUT_SECONDS = 3600
# PoseVAE was trained with this many head-motion styles (0..45).
POSE_STYLES = 46

# One SadTalker process at a time: two would each want ~3 GB of a 6 GB card.
# ponytail: one global lock; a queue of GPU slots if a larger card ever runs two.
_RUN_LOCK = threading.Lock()
# The child prints its measurements as KEY=value lines on stdout; these pick them out.
_REPORT_LINE = re.compile(r"^SADTALKER_(DEVICE|PEAK_VRAM_MB)=(\S+)$", re.MULTILINE)


class SadTalkerError(RuntimeError):
    """SadTalker cannot run or failed; the message says why and how to fix it."""


@dataclass
class SadTalkerResult:
    """What one child run produced: the video file (with audio) and how it ran."""

    video_path: Path
    device: str                 # "cuda" or "cpu", as the child reported it
    peak_vram_mb: Optional[int]  # the child's own peak (None on CPU)
    seconds: float
    # Where the photo sits in the padded square, as fractions of the square's side:
    # (top, left, height, width). Fractions, because the frames come back at 256 px, not the photo's size.
    box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


def missing() -> List[str]:
    """Everything SadTalker still needs, as short phrases naming the fix; empty when it can run."""
    problems: List[str] = []
    if not (SADTALKER_DIR / "inference.py").is_file():
        problems.append(f"the SadTalker code is not checked out at {SADTALKER_DIR} ({FETCH_COMMAND})")
    for relative, (_url, _sha, size) in SADTALKER_FILES.items():
        path = SADTALKER_DIR / relative
        found = path.stat().st_size if path.is_file() else 0
        # Exact size, like VideoSeal: any other size means an interrupted or wrong file.
        if found != size:
            problems.append(
                f"{relative} is {'missing' if not found else f'{found} bytes, expected {size}'} ({FETCH_COMMAND})"
            )
    # find_spec checks that a package is importable without importing it (cheap, no side effects).
    absent = [module for module in SADTALKER_PACKAGES if importlib.util.find_spec(module) is None]
    if absent:
        problems.append(f"Python packages missing: {', '.join(absent)} ({SADTALKER_PIP})")
    return problems


def available() -> bool:
    """True when ``missing()`` finds nothing; cheap (file sizes and import specs only)."""
    return not missing()


def pose_style_for(seed: int) -> int:
    """The head-motion style for a job: a stable choice among the 46 learned styles."""
    return seed % POSE_STYLES


def render(image_rgb: np.ndarray, audio_path: Path, pose_style: int, still: bool) -> SadTalkerResult:
    """
    Animate ``image_rgb`` (the photo as prepared for the render) to ``audio_path``.

    ``still=True`` keeps the head in the photo's pose (expression and mouth only). Returns the
    child's MP4, which lives in a temporary folder the caller must delete with ``cleanup``.
    Raises ``SadTalkerError`` when something is missing or the child fails.
    """
    import cv2

    import gpu_utils

    problems = missing()
    if problems:
        raise SadTalkerError("SadTalker cannot run: " + "; ".join(problems))

    work = Path(tempfile.mkdtemp(prefix="sadtalker-"))
    source = work / "source.png"
    # Pad the short side so the photo is centred in a square (see "Square padding" above).
    # ponytail: the whole square is animated at 256 px, so a small face in a wide or full-body photo
    # comes out soft; crop a head-and-shoulders square around the face and paste it back if such
    # photos become common.
    height, width = image_rgb.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = cv2.copyMakeBorder(image_rgb, top, side - height - top, left, side - width - left, cv2.BORDER_REPLICATE)
    # OpenCV writes BGR; the pipeline's frames are RGB.
    cv2.imwrite(str(source), cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--driven_audio", str(Path(audio_path).resolve()),
        "--source_image", str(source),
        "--result_dir", str(work / "out"),
        "--checkpoint_dir", "checkpoints",
        "--size", "256",
        "--preprocess", "resize",
        "--pose_style", str(pose_style),
    ] + (["--still"] if still else [])
    if not gpu_utils.cuda_available():
        command.append("--cpu")

    started = time.perf_counter()
    with _RUN_LOCK:
        # Unload this process's own models (TTS, watermark) so the child finds the card free.
        gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "SadTalker")
        # PYTHONUNBUFFERED so the child's log lines arrive in order with its report lines.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            done = subprocess.run(
                command, cwd=SADTALKER_DIR, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as err:
            shutil.rmtree(work, ignore_errors=True)
            raise SadTalkerError(f"SadTalker did not finish within {TIMEOUT_SECONDS} s; it was stopped") from err
    seconds = time.perf_counter() - started

    videos = sorted((work / "out").glob("*.mp4"))
    if done.returncode != 0 or not videos:
        # The last lines of stderr hold the exception; the full text goes to the log.
        logger.error("SadTalker failed (exit %s):\n%s\n%s", done.returncode, done.stdout[-4000:], done.stderr[-4000:])
        tail = " | ".join(line for line in done.stderr.strip().splitlines()[-3:]) or "no output video"
        hint = " SadTalker found no face in the photo." if "coeffs of the input" in done.stdout else ""
        shutil.rmtree(work, ignore_errors=True)
        raise SadTalkerError(f"SadTalker failed (exit {done.returncode}): {tail}.{hint}")

    report = dict(_REPORT_LINE.findall(done.stdout))
    peak = report.get("PEAK_VRAM_MB")
    return SadTalkerResult(
        video_path=videos[-1],
        device=report.get("DEVICE", "unknown"),
        peak_vram_mb=int(peak) if peak and peak.isdigit() else None,
        seconds=seconds,
        box=(top / side, left / side, height / side, width / side),
    )


def cleanup(result: SadTalkerResult) -> None:
    """Delete the child's temporary folder (the source PNG, the crop, the video)."""
    # The video sits in <work>/out/, so its grandparent is the folder render() made.
    shutil.rmtree(result.video_path.parent.parent, ignore_errors=True)


def _child_main(argv: List[str]) -> None:
    """
    Run SadTalker's own ``inference.py`` in this process, after four compatibility patches.

    Runs only in the child, with ``SADTALKER_DIR`` as the working directory.
    """
    import runpy
    import types

    # Patch 1: numpy 1.24 removed the alias ``np.float``; one landmark helper still uses it.
    # The alias meant the built-in float, so restoring exactly that is behaviour-preserving.
    np.float = float  # type: ignore[attr-defined]
    # Patch 2: the face *enhancer* module imports gfpgan at load time even when no enhancer is
    # asked for. gfpgan's dependency basicsr cannot import on torchvision 0.20, and no enhancer is
    # used here, so an empty stand-in module satisfies the import. Asking for --enhancer would fail.
    sys.modules["gfpgan"] = types.ModuleType("gfpgan")
    sys.modules["gfpgan"].GFPGANer = None  # type: ignore[attr-defined]
    # Patch 3: SadTalker's code imports ``src.…`` from its own root. sys.path[0] is this file's
    # folder (backend/); replacing it means no backend module can shadow a name SadTalker imports.
    sys.path[0] = str(Path.cwd())
    # Patch 4: align_img builds np.array([w, h, s, t[0], t[1]]) where t[0] and t[1] are 1-element
    # arrays; numpy 1.24+ refuses that mixed ("inhomogeneous") list. The copy below differs only in
    # taking the scalars out with .item(). It replaces the name the preprocessor calls at run time.
    # These modules exist only inside the SadTalker checkout (the child's working directory), so the
    # type checker, which reads the repository, cannot see them.
    import src.utils.preprocess as crop_module  # pyrefly: ignore[missing-import]
    from src.face3d.util import preprocess as face3d  # pyrefly: ignore[missing-import]

    def align_img(img, lm, lm3D, mask=None, target_size=224.0, rescale_factor=102.0):
        width, height = img.size
        lm5p = face3d.extract_5p(lm) if lm.shape[0] != 5 else lm
        t, s = face3d.POS(lm5p.transpose(), lm3D.transpose())
        s = rescale_factor / s
        img_new, lm_new, mask_new = face3d.resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
        return np.array([width, height, s, t[0].item(), t[1].item()]), img_new, lm_new, mask_new

    crop_module.align_img = align_img

    # inference.py is a script: run it as __main__ with our arguments, as if from its command line.
    sys.argv = ["inference.py", *argv]
    runpy.run_path("inference.py", run_name="__main__")

    import torch

    device = "cuda" if torch.cuda.is_available() and "--cpu" not in argv else "cpu"
    print(f"SADTALKER_DEVICE={device}")
    if device == "cuda":
        # Reserved, not allocated: what the card actually had to hold for this process.
        print(f"SADTALKER_PEAK_VRAM_MB={torch.cuda.max_memory_reserved() // (1024 * 1024)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_main(sys.argv[2:])
    else:
        # A quick status check: PYTHONPATH=backend python backend/sadtalker_engine.py
        print(json.dumps({"available": available(), "missing": missing()}, indent=2))
