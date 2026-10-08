"""
Render worker: ``AvatarRenderJob`` -> MP4 (tasks G2-05 / G2-06).

This is the vision side of the frozen contract. Everything it needs arrives in
the job: which registered avatar, which audio file, the phoneme timestamps,
the emotion vector, the quality tier and the frame rate. The steps are

    avatar image (consent checked) -> landmarks -> rig
    timestamps + audio loudness    -> per-frame blendshape weights
    weights -> frames -> (optional Wav2Lip mouth) -> H.264 + AAC MP4

Two engines, chosen by name and reported by name in the result:

* ``blendshape`` -- CPU mesh warp driven by the viseme timeline. Always
  available, needs no GPU.
* ``wav2lip`` -- neural mouth from the audio itself; the warp still supplies
  blinks and brows. Needs the licence-gated checkpoint.

There is no fallback between them. A job that asks for ``wav2lip`` without the
checkpoint fails with the command that fetches it; it does not quietly come
back as a blendshape render.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

import audit_log
import gpu_utils
import manifest as manifest_module
import video_io
import video_watermark
import watermark_engine
from avatar_store import AvatarConsentError, AvatarStore
from contracts import AvatarRenderJob, BackgroundSpec, RenderQuality
from face_animation import (
    AnimationTrack,
    build_animation,
    frame_count_for,
    seed_from_job_id,
    speech_energy_envelope,
)
from face_engine import FACE_ENGINE_LOCK, FaceAnalysis, shared_face_engine, shared_segmenter
from face_warp import PortraitAnimator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INPUTS_DIR = PROJECT_ROOT / "inputs"
RENDERS_DIR = OUTPUTS_DIR / "renders"

ENGINE_BLENDSHAPE = "blendshape"
ENGINE_WAV2LIP = "wav2lip"
ENGINES = (ENGINE_BLENDSHAPE, ENGINE_WAV2LIP)

# The largest frame each quality tier may produce. A photo is never enlarged
# to reach it (see ``video_io.fit_within``).
QUALITY_BOX: Dict[RenderQuality, Tuple[int, int]] = {
    RenderQuality.PREVIEW: (512, 512),
    RenderQuality.HD_1080P: (1920, 1080),
}

AI_LABEL = "AI-generated"
# Audio and contract duration may differ by this much before it is reported.
DURATION_TOLERANCE_S = 0.12

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_MOUTH_PREFIXES = ("jaw", "mouth", "tongue")

ProgressCallback = Callable[[int, int], None]


class RenderError(RuntimeError):
    """The job cannot be rendered; the message says why and how to fix it."""


def default_engine() -> str:
    """The engine used when a job does not name one (``RENDER_ENGINE`` in .env)."""
    return os.getenv("RENDER_ENGINE", ENGINE_BLENDSHAPE).strip().lower() or ENGINE_BLENDSHAPE


def validate_engine(engine: Optional[str]) -> str:
    name = (engine or default_engine()).strip().lower()
    if name not in ENGINES:
        raise RenderError(f"unknown render engine {name!r}; choose one of {list(ENGINES)}")
    return name


def resolve_audio_url(audio_url: str) -> Path:
    """Map the job's ``audioUrl`` to a file inside ``outputs/`` or ``inputs/``."""
    return _resolve_media_url(
        audio_url,
        field="audioUrl",
        noun="audio",
        hint="Use the /outputs/ URL returned by POST /api/v1/audio/synthesize",
    )


def _resolve_media_url(url: str, field: str, noun: str, hint: str) -> Path:
    """
    Map a job URL to a file inside ``outputs/`` or ``inputs/``.

    Accepted: ``file://`` URLs, and ``http(s)`` URLs whose path is under
    ``/outputs/`` -- which is how this API serves its own generated audio, so
    the URL the frontend plays is the URL it can submit. Anything else
    (``s3://``, another host's path) is refused: the worker does not fetch
    remote media, and must not be talked into reading arbitrary local files.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    if parsed.scheme == "file":
        candidate = Path(path)
    elif parsed.scheme in ("http", "https") and path.startswith("/outputs/"):
        candidate = OUTPUTS_DIR / path[len("/outputs/"):]
    else:
        raise RenderError(
            f"{field} {url!r} is not renderable here. {hint}, or a file:// URL to a file "
            "in the project's outputs/ or inputs/ folder."
        )
    try:
        resolved = candidate.resolve()
    except (ValueError, OSError) as err:  # e.g. an embedded NUL byte
        raise RenderError(f"{field} is not a usable path: {err}") from err
    roots = (OUTPUTS_DIR.resolve(), INPUTS_DIR.resolve())
    if not any(root in resolved.parents for root in roots):
        raise RenderError(f"{field} must point inside the project's outputs/ or inputs/ folder")
    if not resolved.is_file():
        raise RenderError(f"{noun} file not found: {resolved.name}. {hint} that exists.")
    return resolved


def output_path_for(job_id: str) -> Path:
    """
    ``outputs/renders/<jobId>.mp4`` with the id reduced to a safe filename.

    An id that is already a safe filename is used as-is. Any other id gets a
    hash of the original appended, so two different jobs ("a b" and "a_b")
    can never be given the same file.
    """
    name = _SAFE_NAME.sub("_", job_id).strip("_")[:80]
    if name != job_id:
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:10]
        name = f"{name or 'job'}-{digest}"
    return RENDERS_DIR / f"{name}.mp4"


def output_size(width: int, height: int, quality: RenderQuality) -> Tuple[int, int]:
    box = QUALITY_BOX[quality]
    return video_io.fit_within(width, height, box[0], box[1])


@dataclass
class RenderResult:
    """What a finished render produced, and how."""

    job_id: str
    avatar_id: str
    engine: str
    output_path: str
    video_url: str
    width: int
    height: int
    fps: int
    frame_count: int
    duration_seconds: float
    render_seconds: float
    media: Dict[str, object] = field(default_factory=dict)
    energy_gated: bool = False
    blink_count: int = 0
    unknown_visemes: Dict[str, int] = field(default_factory=dict)
    peak_vram_mb: Optional[int] = None
    background: Optional[str] = None
    # Set when a small photo was enlarged with super-resolution to reach the 1080P_HQ box (T8.8).
    super_resolution: Optional[Dict[str, object]] = None
    # What the invisible video watermark did and where the signed manifest is (both always present).
    watermark: Optional[Dict[str, object]] = None
    manifest: Optional[Dict[str, object]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        """Render time per second of video; below 1.0 is faster than real time."""
        return self.render_seconds / self.duration_seconds if self.duration_seconds else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "jobId": self.job_id,
            "avatarId": self.avatar_id,
            "engine": self.engine,
            "outputPath": self.output_path,
            "videoUrl": self.video_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frameCount": self.frame_count,
            "durationSeconds": round(self.duration_seconds, 3),
            "renderSeconds": round(self.render_seconds, 3),
            "realtimeFactor": round(self.realtime_factor, 3),
            "media": self.media,
            "energyGated": self.energy_gated,
            "blinkCount": self.blink_count,
            "unknownVisemes": self.unknown_visemes,
            "peakVramMb": self.peak_vram_mb,
            "background": self.background,
            "superResolution": self.super_resolution,
            "watermark": self.watermark,
            "manifest": self.manifest,
            "warnings": self.warnings,
        }


def preflight(job: AvatarRenderJob, engine: Optional[str] = None, store: Optional[AvatarStore] = None) -> Path:
    """
    Cheap checks that can reject a job before it is queued.

    Raises ``AvatarNotFound`` / ``AvatarConsentError`` / ``RenderError``.
    Returns the resolved audio path. Loads no model.
    """
    name = validate_engine(engine)
    try:
        (store or AvatarStore()).require_usable(job.avatar_id)
    except AvatarConsentError as err:
        audit_log.shared_audit().record("face_refused", subject=job.avatar_id, basis=None, reason=str(err), job=job.job_id)
        raise
    audio = resolve_audio_url(job.audio_url)
    if job.background is not None:
        _check_background(job.background)
    if watermark_engine.enabled() and not video_watermark.VideoWatermarker.files_present():
        from model_registry import VISION_FETCH_COMMAND

        raise RenderError(
            "the invisible video watermark model is missing, and every rendered video must carry the mark. "
            f"Fetch it with: {VISION_FETCH_COMMAND} --only videoseal (set WATERMARK_ENABLED=false to render "
            "unmarked, which the result will say)"
        )
    if name == ENGINE_WAV2LIP:
        from wav2lip_engine import shared_wav2lip_engine

        if not shared_wav2lip_engine().available:
            from model_registry import VISION_FETCH_COMMAND

            raise RenderError(
                "the 'wav2lip' engine has no checkpoint. It is under a research / "
                f"non-commercial licence, so a person has to opt in: {VISION_FETCH_COMMAND} "
                "--only wav2lip --accept-licence wav2lip. Or render with engine 'blendshape'."
            )
    return audio


def _check_background(spec: BackgroundSpec) -> None:
    """Refuse a background request that cannot be honoured, before queueing."""
    if not shared_segmenter().available:
        from model_registry import VISION_FETCH_COMMAND

        raise RenderError(
            "a background was requested but the selfie segmenter model is missing, and the "
            f"render would silently keep the old background. Fetch it with: {VISION_FETCH_COMMAND}"
        )
    if spec.image_url is not None:
        _background_image_path(spec)


def _background_image_path(spec: BackgroundSpec) -> Path:
    return _resolve_media_url(
        spec.image_url or "",
        field="background.imageUrl",
        noun="background image",
        hint="Use the /outputs/ URL of an image",
    )


def _cover_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``image`` to fill width x height, cropping the overflow (no stretching)."""
    import cv2

    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(width, round(image.shape[1] * scale)), max(height, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


# A person who fills less than this fraction of the frame means the segmenter
# probably missed them; the new background would then replace the subject too.
MIN_FOREGROUND_FRACTION = 0.03


def apply_background(image: np.ndarray, spec: BackgroundSpec) -> Tuple[np.ndarray, List[str]]:
    """
    Composite the photo's subject over the requested background.

    Done once on the source photo, before the face is analysed and warped,
    rather than on every frame: the warp moves pixels only near the eyes,
    brows and mouth, so a background baked into the photo stays still, costs
    nothing per frame, and no pixel of the old background can be dragged in.
    """
    background: Tuple[int, int, int] | np.ndarray
    if spec.color is not None:
        background = (int(spec.color[1:3], 16), int(spec.color[3:5], 16), int(spec.color[5:7], 16))
    else:
        import cv2

        path = _background_image_path(spec)
        loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise RenderError(f"background image {path.name} could not be decoded as an image")
        background = _cover_fit(cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB), image.shape[1], image.shape[0])

    warnings: List[str] = []
    with FACE_ENGINE_LOCK:
        segmenter = shared_segmenter()
        mask = segmenter.foreground_mask(image)
        if float((mask > 0.5).mean()) < MIN_FOREGROUND_FRACTION:
            warnings.append(
                "the segmenter found almost no person in this photo, so the background "
                "replacement may have covered the subject"
            )
        composited = segmenter.replace_background(image, background)
    return composited, warnings


def _describe_background(spec: Optional[BackgroundSpec]) -> Optional[str]:
    """What the result reports about the background, so a replaced one is never silent."""
    if spec is None:
        return None
    return f"color {spec.color}" if spec.color else f"image {Path(spec.image_url or '').name}"


def _verify_video_mark(path: Path, manifest_id: bytes) -> Dict[str, object]:
    """
    Read the mark back from the encoded file and insist it is there, with this video's id.

    Checked on the file as written (after H.264), not on the frames before it, because the
    encoding is what the mark has to survive. A video that would claim a mark it does not carry
    is not delivered (golden rule 1).
    """
    report = video_watermark.shared_video_watermarker().detect_video(path, sample_frames=16)
    if not report.detected:
        raise RenderError(
            f"the video watermark was embedded but could not be read back from the encoded file "
            f"({report.to_dict()}); refusing to deliver a video that claims a mark it does not carry"
        )
    return {"applied": True, "verified": True, "embeddedManifestId": manifest_id.hex(), **report.to_dict()}


def _issue_manifest(manifest_id: bytes, video: Path, audio: Path, avatar: Dict[str, object],
                    watermark: Dict[str, object], render: Dict[str, object]) -> Dict[str, object]:
    """Write the signed manifest beside the video and say where it is."""
    models = {
        "speech": (manifest_module.read_speech_record(audio).get("model")),
        "renderer": render.get("engine"),
        "landmarks": "MediaPipe Face Landmarker (478 points)",
        "audioWatermark": watermark_engine.METHOD if watermark_engine.enabled() else None,
        "videoWatermark": video_watermark.METHOD if watermark_engine.enabled() else None,
    }
    document = manifest_module.build_manifest(
        manifest_id=manifest_id, video_path=video, audio_path=audio, avatar=avatar, render=render,
        video_watermark=watermark, models=models,
    )
    path = Path(str(video) + ".manifest.json")
    path.write_text(json.dumps(document, indent=2, sort_keys=True))
    try:
        url = f"/outputs/{path.resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()}"
    except ValueError:
        url = path.resolve().as_uri()
    return {"manifestId": manifest_id.hex(), "path": str(path), "url": url, "videoSha256": document["content"]["videoSha256"]}


MIN_HD_SIDE = 720  # a 1080P_HQ render below this short side gets the photo super-resolved first


def _maybe_super_resolve(
    image: np.ndarray, quality: RenderQuality, warnings: List[str]
) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    """
    For 1080P_HQ, enlarge a photo that would give less than 720p with super-resolution (once, before animation).

    A photo is still never stretched by plain resizing: either the network enlarges it, and the
    result says the added detail is synthesised, or the video stays at the photo's size and a
    warning says why and how to fix it. PREVIEW never enlarges.
    """
    import super_resolution

    height, width = image.shape[:2]
    # Only photos that would otherwise give less than 720p are enlarged: lifting a 1024 px photo to
    # 1080 is a 5% gain for a 4x network pass, and not worth inventing detail for.
    if quality != RenderQuality.HD_1080P or min(output_size(width, height, quality)) >= MIN_HD_SIDE:
        return image, None
    resolver = super_resolution.shared_resolver()
    if not resolver.available:
        warnings.append(
            f"1080P_HQ was asked for but the photo is {width}x{height} and the super-resolution weights are missing, "
            "so the video stays at the photo's size. Fetch them with: "
            "PYTHONPATH=backend backend/.conda/bin/python scripts/fetch_vision_models.py --only realesrgan"
        )
        return image, None
    started = time.perf_counter()
    enlarged = resolver.upscale(image)
    warnings.append("the photo was enlarged with super-resolution; the added fine detail is synthesised, not recovered")
    return enlarged, {
        "method": super_resolution.METHOD, "photo": [width, height], "enlarged": [enlarged.shape[1], enlarged.shape[0]],
        "seconds": round(time.perf_counter() - started, 2),
    }


def _stamp_label(frame: np.ndarray) -> np.ndarray:
    """Burn the disclosure label into the bottom-left corner (task FD-03)."""
    import cv2

    height, width = frame.shape[:2]
    scale = max(0.4, width / 1100.0)
    thickness = max(1, int(round(scale * 1.6)))
    origin = (int(width * 0.025), height - int(height * 0.03))
    cv2.putText(frame, AI_LABEL, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, AI_LABEL, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _warp_frames(
    animator: PortraitAnimator, track: AnimationTrack, include_mouth: bool
) -> Iterator[np.ndarray]:
    for index in range(track.frame_count):
        weights = track.frame(index)
        if not include_mouth:
            weights = {k: v for k, v in weights.items() if not k.startswith(_MOUTH_PREFIXES)}
        yield animator.render(weights)


def render_job(
    job: AvatarRenderJob,
    engine: Optional[str] = None,
    store: Optional[AvatarStore] = None,
    output_path: Optional[Path] = None,
    label: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> RenderResult:
    """Render one job to an MP4 and describe the result."""
    import cv2

    started = time.perf_counter()
    engine_name = validate_engine(engine)
    store = store or AvatarStore()
    audio_path = preflight(job, engine_name, store)
    warnings: List[str] = []

    image = store.load_image(job.avatar_id)
    image, super_res = _maybe_super_resolve(image, job.render_quality, warnings)
    width, height = output_size(image.shape[1], image.shape[0], job.render_quality)
    if (width, height) != (image.shape[1], image.shape[0]):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if job.background is not None:
        image, background_warnings = apply_background(image, job.background)
        warnings.extend(background_warnings)

    # Analysed at output size, so the rig's landmarks are in output pixels.
    with FACE_ENGINE_LOCK:
        analysis: FaceAnalysis = shared_face_engine().analyze(image)
    animator = PortraitAnimator(image, analysis)

    import soundfile as sf

    audio_info = sf.info(str(audio_path))
    audio_seconds = audio_info.frames / float(audio_info.samplerate)
    if abs(audio_seconds - job.duration_seconds) > DURATION_TOLERANCE_S:
        warnings.append(
            f"durationSeconds is {job.duration_seconds:.2f} but the audio is "
            f"{audio_seconds:.2f} s long; the video follows the audio"
        )
    duration = audio_seconds

    track = build_animation(
        job.phoneme_timestamps,
        duration_seconds=duration,
        fps=job.target_fps,
        emotion_vector=job.emotion_vector.model_dump(by_alias=True, exclude_none=True),
        seed=seed_from_job_id(job.job_id),
        energy_envelope=speech_energy_envelope(audio_path),
    )
    if track.unknown_visemes:
        warnings.append(
            "visemes outside the 15-viseme contract were rendered as the rest pose: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(track.unknown_visemes.items()))
        )

    total = track.frame_count
    assert total == frame_count_for(duration, job.target_fps)
    peak_vram: Optional[int] = None
    if engine_name == ENGINE_WAV2LIP:
        from wav2lip_engine import face_box_from_bbox, shared_wav2lip_engine

        box = analysis.bounding_box
        face_box = face_box_from_bbox(box.x, box.y, box.width, box.height, width, height)
        gpu_utils.reset_peak()
        frames: Iterator[np.ndarray] = shared_wav2lip_engine().sync_frames(
            _warp_frames(animator, track, include_mouth=False),
            face_box,
            audio_path,
            fps=job.target_fps,
            frame_count=total,
        )
    else:
        frames = _warp_frames(animator, track, include_mouth=True)

    # The invisible mark goes in before the visible label, one window of frames at a time as they are
    # encoded. Its message names this video's manifest, which is issued below once the file exists.
    manifest_id = manifest_module.new_manifest_id()
    marking = watermark_engine.enabled()
    if marking:
        frames = video_watermark.shared_video_watermarker().embed_stream(frames, manifest_id)

    destination = Path(output_path) if output_path else output_path_for(job.job_id)
    # Encoded beside the destination and moved into place only when complete,
    # so a failed render never truncates or deletes an earlier finished video.
    partial = destination.with_name(destination.stem + ".partial.mp4")
    watermark_result: Dict[str, object]
    try:
        with video_io.VideoWriter(
            # The cutoff is the end of the LAST FRAME, not the end of the audio: with
            # the audio's length, ffmpeg dropped a final frame that started within
            # ~10 ms of it (3.25 s at 25 fps: 82 frames rendered, 81 kept). The
            # audio itself is never cut: it is shorter than the cutoff.
            partial, width, height, job.target_fps, audio_path=audio_path,
            duration=max(duration, total / job.target_fps),
        ) as writer:
            for index, frame in enumerate(frames):
                writer.write(_stamp_label(frame) if label else frame)
                if progress is not None and (index % 10 == 0 or index + 1 == total):
                    progress(index + 1, total)
        if engine_name == ENGINE_WAV2LIP:
            peak_vram = gpu_utils.peak_vram_mb()

        media = video_io.probe(partial)
        if not (media.has_video and media.has_audio):
            raise RenderError(
                f"the encoder produced a file with a missing stream ({media.to_dict()}); "
                "the clip may be shorter than one frame at this targetFps"
            )
        watermark_result = _verify_video_mark(partial, manifest_id) if marking else {
            "applied": False, "reason": "disabled by WATERMARK_ENABLED=false"}
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    if media.frame_count and media.frame_count != total:
        warnings.append(f"{total} frames were rendered but the file holds {media.frame_count}")
        total = media.frame_count

    try:
        relative = destination.resolve().relative_to(OUTPUTS_DIR.resolve())
        video_url = f"/outputs/{relative.as_posix()}"
    except ValueError:
        video_url = destination.resolve().as_uri()

    manifest_info = _issue_manifest(
        manifest_id, destination, audio_path, store.get(job.avatar_id).to_dict(), watermark_result,
        render={
            "width": width, "height": height, "fps": job.target_fps, "frameCount": total,
            "durationSeconds": round(duration, 3), "engine": engine_name, "quality": job.render_quality.value,
            "background": _describe_background(job.background), "label": bool(label),
            "superResolution": super_res,
        },
    )

    avatar_provenance = store.get(job.avatar_id).provenance
    audit = audit_log.shared_audit()
    audit.record(
        "manifest_issued", subject=str(manifest_info["manifestId"]), basis=None, video_sha256=manifest_info["videoSha256"],
        job=job.job_id, avatar=job.avatar_id, watermarked=bool(watermark_result.get("applied")),
    )
    audit.record(
        "face_use", subject=job.avatar_id, basis=str(avatar_provenance.get("consentBasis") or avatar_provenance.get("source") or ""),
        source=avatar_provenance.get("source"), job=job.job_id, engine=engine_name, manifest=manifest_info["manifestId"],
    )

    result = RenderResult(
        job_id=job.job_id,
        avatar_id=job.avatar_id,
        engine=engine_name,
        output_path=str(destination),
        video_url=video_url,
        width=width,
        height=height,
        fps=job.target_fps,
        frame_count=total,
        duration_seconds=duration,
        render_seconds=time.perf_counter() - started,
        media=media.to_dict(),
        energy_gated=track.energy_gated,
        blink_count=len(track.blink_times),
        unknown_visemes=track.unknown_visemes,
        peak_vram_mb=peak_vram,
        background=_describe_background(job.background),
        super_resolution=super_res,
        watermark=watermark_result,
        manifest=manifest_info,
        warnings=warnings,
    )
    logger.info(
        "Rendered %s: %s engine, %d frames %dx%d in %.2fs (%.2fx real time)",
        job.job_id, engine_name, total, width, height, result.render_seconds, result.realtime_factor,
    )
    return result
