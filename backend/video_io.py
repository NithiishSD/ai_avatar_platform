"""
Video plumbing: encode frames + audio to MP4, and read them back (task G2-04).

Everything goes through the system ``ffmpeg`` / ``ffprobe`` binaries rather
than a Python wrapper, for two reasons. OpenCV's ``VideoWriter`` cannot mux an
audio stream at all, and its H.264 support depends on how the wheel was built.
And piping raw frames straight into ffmpeg means no intermediate image files:
a 60 second 1080p render never touches the disk until it is a finished MP4.

The output is H.264 / yuv420p / AAC with ``+faststart``, which is the one
combination every browser's ``<video>`` element will play from a static file
server without a second round trip for the index.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install it with: sudo apt install ffmpeg"


class FFmpegNotFound(RuntimeError):
    """``ffmpeg`` or ``ffprobe`` is not on PATH."""


class VideoEncodeError(RuntimeError):
    """ffmpeg exited with an error; the message carries its stderr."""


def require_binary(name: str) -> str:
    """Absolute path of ``ffmpeg`` / ``ffprobe``, or an actionable error."""
    path = shutil.which(name)
    if path is None:
        raise FFmpegNotFound(f"{name} was not found on PATH. {INSTALL_HINT}")
    return path


def even(value: float) -> int:
    """Round down to an even integer: yuv420p cannot encode odd dimensions."""
    return max(2, int(value) // 2 * 2)


def fit_within(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """
    Scale ``width x height`` down to fit the box, preserving aspect ratio.

    Never scales up: enlarging a photo adds pixels, not detail, and would let
    a 512 px render be labelled "1080p".
    """
    scale = min(1.0, max_width / width, max_height / height)
    return even(width * scale), even(height * scale)


class VideoWriter:
    """
    Stream RGB frames into an MP4, optionally muxed with an audio file.

        with VideoWriter(path, width, height, fps, audio_path=wav) as writer:
            for frame in frames:
                writer.write(frame)

    ``-shortest`` trims whichever stream runs long, so the container never
    ends on a frozen frame or a tail of silence; the two streams end up within
    one frame of each other.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: int,
        audio_path: Optional[Union[str, Path]] = None,
        crf: int = 20,
        preset: str = "veryfast",
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError(
                f"video dimensions must be even for yuv420p, got {width}x{height}"
            )
        self.output_path = Path(output_path)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.audio_path = Path(audio_path) if audio_path else None
        self.crf, self.preset = int(crf), preset
        self.frames_written = 0
        self._process: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        cmd = [
            require_binary("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
        ]
        if self.audio_path is not None:
            cmd += ["-i", str(self.audio_path), "-map", "0:v:0", "-map", "1:a:0"]
        cmd += [
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
        ]
        if self.audio_path is not None:
            cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-movflags", "+faststart", str(self.output_path)]
        return cmd

    def __enter__(self) -> "VideoWriter":
        if self.audio_path is not None and not self.audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {self.audio_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("VideoWriter used outside its 'with' block")
        if frame_rgb.shape[:2] != (self.height, self.width) or frame_rgb.shape[2] != 3:
            raise ValueError(
                f"frame is {frame_rgb.shape}, expected ({self.height}, {self.width}, 3)"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes())
        except BrokenPipeError as err:
            raise VideoEncodeError(self._finish() or "ffmpeg closed its input early") from err
        self.frames_written += 1

    def _finish(self) -> str:
        """Close the pipe, wait for ffmpeg, return its stderr."""
        process = self._process
        if process is None:
            return ""
        self._process = None
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            process.stderr.close()
        code = process.wait()
        if code != 0:
            return stderr.strip() or f"ffmpeg exited with code {code}"
        return ""

    def __exit__(self, exc_type, exc, tb) -> None:
        error = self._finish()
        if exc_type is not None:
            # The caller failed mid-render: do not leave a truncated MP4 that
            # looks like a finished one.
            self.output_path.unlink(missing_ok=True)
            return
        if error:
            self.output_path.unlink(missing_ok=True)
            raise VideoEncodeError(f"ffmpeg failed writing {self.output_path.name}: {error}")
        if self.frames_written == 0:
            self.output_path.unlink(missing_ok=True)
            raise VideoEncodeError("no frames were written")


@dataclass(frozen=True)
class MediaInfo:
    """What ``ffprobe`` reports about a rendered file."""

    path: str
    has_video: bool
    has_audio: bool
    width: int
    height: int
    fps: float
    frame_count: int
    video_duration: float
    audio_duration: float
    video_codec: str
    audio_codec: str

    @property
    def duration_gap(self) -> float:
        """Absolute difference between the two stream durations, in seconds."""
        if not (self.has_video and self.has_audio):
            return 0.0
        return abs(self.video_duration - self.audio_duration)

    def to_dict(self) -> Dict[str, object]:
        return {
            "hasVideo": self.has_video,
            "hasAudio": self.has_audio,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frameCount": self.frame_count,
            "videoDuration": round(self.video_duration, 3),
            "audioDuration": round(self.audio_duration, 3),
            "durationGap": round(self.duration_gap, 3),
            "videoCodec": self.video_codec,
            "audioCodec": self.audio_codec,
        }


def _rate(value: str) -> float:
    try:
        numerator, _, denominator = value.partition("/")
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def parse_probe(path: Union[str, Path], payload: Dict[str, object]) -> MediaInfo:
    """Build a ``MediaInfo`` from ffprobe's JSON (split out for testing)."""
    streams = payload.get("streams") or []
    container = _float((payload.get("format") or {}).get("duration"))  # type: ignore[union-attr]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)  # type: ignore[union-attr]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)  # type: ignore[union-attr]
    frames = 0
    if video is not None:
        try:
            frames = int(video.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frames = 0
    return MediaInfo(
        path=str(path),
        has_video=video is not None,
        has_audio=audio is not None,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        fps=_rate(str(video.get("avg_frame_rate", "0/1"))) if video else 0.0,
        frame_count=frames,
        video_duration=(_float(video.get("duration")) or container) if video else 0.0,
        audio_duration=(_float(audio.get("duration")) or container) if audio else 0.0,
        video_codec=str(video.get("codec_name", "")) if video else "",
        audio_codec=str(audio.get("codec_name", "")) if audio else "",
    )


def probe(path: Union[str, Path]) -> MediaInfo:
    """Inspect a media file with ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path}")
    result = subprocess.run(
        [
            require_binary("ffprobe"),
            "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoEncodeError(f"ffprobe could not read {path.name}: {result.stderr.strip()}")
    return parse_probe(path, json.loads(result.stdout or "{}"))


def read_frames(
    path: Union[str, Path], fps: Optional[float] = None
) -> Iterator[np.ndarray]:
    """
    Yield the RGB frames of a video, optionally resampled to ``fps``.

    Decoded by ffmpeg into a raw pipe, so the frame rate conversion the
    lip-sync metric needs (SyncNet was trained at 25 fps) is done by the same
    tool that did the encoding.
    """
    info = probe(path)
    if not info.has_video:
        raise VideoEncodeError(f"{Path(path).name} has no video stream")
    cmd = [require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = info.width * info.height * 3
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(info.height, info.width, 3)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.wait()


def read_audio(path: Union[str, Path], sample_rate: int = 16000) -> np.ndarray:
    """Decode any media file's audio to mono float32 in [-1, 1]."""
    result = subprocess.run(
        [
            require_binary("ffmpeg"),
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoEncodeError(
            f"ffmpeg could not decode audio from {Path(path).name}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
