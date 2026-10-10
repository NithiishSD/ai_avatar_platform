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

Where it sits: ``render_engine`` writes every finished render through ``VideoWriter``; ``probe``,
``read_frames`` and ``read_audio`` are used by the metrics, the watermark detectors, the
authenticity check and voice-to-avatar uploads to read files back.

Concepts used below, explained once:

**A pipe** connects one process's output to another's input. ``subprocess.Popen(..., stdin=PIPE)``
starts ffmpeg and hands us a file-like object; every byte we write to it, ffmpeg reads as its input
file (``-i -`` means "read from standard input"). Reading works the same way in reverse with
``stdout=PIPE`` and an output name of ``-``.

**Raw video** (``-f rawvideo -pix_fmt rgb24``) is just pixels with no header: width x height x 3
bytes per frame, red-green-blue, row by row. Because there is no header, ffmpeg must be told the
size and frame rate up front, and a reader knows a frame is complete after exactly that many bytes.

**Muxing** puts separately encoded streams (video, audio) into one container file (MP4).

**H.264 / yuv420p**: H.264 is the video codec. yuv420p stores brightness for every pixel but colour
only once per 2x2 block, which halves the data and is the only layout every player supports. It is
also why width and height must be even.
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

# Appended to every "not found" error so it says how to fix itself (golden rule 7).
INSTALL_HINT = "Install it with: sudo apt install ffmpeg"


class FFmpegNotFound(RuntimeError):
    """``ffmpeg`` or ``ffprobe`` is not on PATH."""


class VideoEncodeError(RuntimeError):
    """ffmpeg exited with an error; the message carries its stderr."""


def require_binary(name: str) -> str:
    """Absolute path of ``ffmpeg`` / ``ffprobe``, or an actionable error."""
    # shutil.which searches PATH the same way a shell would, without running anything.
    path = shutil.which(name)
    if path is None:
        raise FFmpegNotFound(f"{name} was not found on PATH. {INSTALL_HINT}")
    return path


def even(value: float) -> int:
    """Round down to an even integer: yuv420p cannot encode odd dimensions."""
    # // 2 * 2 drops the last bit (7 -> 6); max(2, ...) keeps a degenerate input from becoming 0.
    return max(2, int(value) // 2 * 2)


def fit_within(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """
    Scale ``width x height`` down to fit the box, preserving aspect ratio.

    Never scales up: enlarging a photo adds pixels, not detail, and would let
    a 512 px render be labelled "1080p".
    """
    # The smaller ratio is the one that fits both sides; capping at 1.0 is the "never scale up" rule.
    scale = min(1.0, max_width / width, max_height / height)
    return even(width * scale), even(height * scale)


class VideoWriter:
    """
    Stream RGB frames into an MP4, optionally muxed with an audio file.

        with VideoWriter(path, width, height, fps, audio_path=wav) as writer:
            for frame in frames:
                writer.write(frame)

    Pass ``duration`` (the audio's length) and the output is cut to exactly
    that: the caller writes enough frames to cover it, and no frame or audio
    is dropped. Without it ``-shortest`` trims whichever stream runs long.
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
        duration: Optional[float] = None,
    ) -> None:
        # Checked here rather than left to ffmpeg, whose error for odd sizes arrives only after the
        # first frame is piped and is harder to read.
        if width % 2 or height % 2:
            raise ValueError(
                f"video dimensions must be even for yuv420p, got {width}x{height}"
            )
        self.output_path = Path(output_path)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.audio_path = Path(audio_path) if audio_path else None
        # CRF (constant rate factor) is H.264's quality dial: lower is better and bigger, 18-23 is the
        # usual range. The preset trades encode speed for file size; "veryfast" favours speed.
        self.crf, self.preset = int(crf), preset
        self.duration = float(duration) if duration else None
        # Counted so __exit__ can refuse to leave an empty MP4 behind.
        self.frames_written = 0
        # The running ffmpeg process; it exists only inside the ``with`` block.
        self._process: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        """
        The ffmpeg argument list for this writer (split out so tests can inspect it without ffmpeg).

        ffmpeg options apply to the *next* ``-i`` input or, after the last input, to the output, so
        the order of this list matters.
        """
        # A list, never a shell string: no quoting problems and no shell injection through a path.
        cmd = [
            require_binary("ffmpeg"),
            "-y",  # overwrite an existing output instead of stopping to ask
            "-hide_banner",
            "-loglevel", "error",  # only real errors reach stderr, so stderr is the error message
            "-f", "rawvideo",  # input 0 is headerless pixels...
            "-pix_fmt", "rgb24",  # ...3 bytes per pixel, in the order numpy/PIL hold them
            "-s", f"{self.width}x{self.height}",  # raw video has no header, so the size is given here
            "-r", str(self.fps),
            "-i", "-",  # read input 0 from our pipe
        ]
        if self.audio_path is not None:
            # Input 1 is the audio file. -map picks exactly the first video stream of input 0 and the
            # first audio stream of input 1, so nothing else in the audio file sneaks into the MP4.
            cmd += ["-i", str(self.audio_path), "-map", "0:v:0", "-map", "1:a:0"]
        cmd += [
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",  # convert from rgb24 to the layout every player decodes
        ]
        if self.audio_path is not None:
            # AAC is the audio codec browsers expect inside MP4; 160 kbit/s is transparent for speech.
            cmd += ["-c:a", "aac", "-b:a", "160k"]
            # -t cuts to an exact duration (microsecond precision); -shortest stops at the shorter stream.
            cmd += ["-t", f"{self.duration:.6f}"] if self.duration else ["-shortest"]
        # +faststart moves the MP4 index to the front so a browser can start playing before the end.
        cmd += ["-movflags", "+faststart", str(self.output_path)]
        return cmd

    # __enter__/__exit__ make this a context manager: ``with VideoWriter(...) as w`` runs __enter__
    # first and guarantees __exit__ runs afterwards, even if the body raises.
    def __enter__(self) -> "VideoWriter":
        """Start ffmpeg and return the writer. Raises ``FileNotFoundError`` for a missing audio file."""
        # Checked before starting ffmpeg, so the error names the file instead of quoting ffmpeg.
        if self.audio_path is not None and not self.audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {self.audio_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,  # we write frames here
            stdout=subprocess.DEVNULL,  # the MP4 goes to a file, so stdout carries nothing we need
            stderr=subprocess.PIPE,  # kept, to put ffmpeg's own words in our error message
        )
        return self

    def write(self, frame_rgb: np.ndarray) -> None:
        """
        Send one ``(height, width, 3)`` RGB frame to ffmpeg.

        Raises ``ValueError`` for a wrongly shaped frame and ``VideoEncodeError`` if ffmpeg has died.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("VideoWriter used outside its 'with' block")
        # A wrong size would not fail in ffmpeg: it would silently shear every later frame, because raw
        # video is cut into frames purely by byte count. So the shape is checked on every write.
        if frame_rgb.shape[:2] != (self.height, self.width) or frame_rgb.shape[2] != 3:
            raise ValueError(
                f"frame is {frame_rgb.shape}, expected ({self.height}, {self.width}, 3)"
            )
        # tobytes() of a contiguous uint8 array is exactly the rgb24 layout ffmpeg was promised.
        try:
            self._process.stdin.write(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes())
        except BrokenPipeError as err:
            # A broken pipe means ffmpeg exited; its stderr (collected by _finish) says why.
            raise VideoEncodeError(self._finish() or "ffmpeg closed its input early") from err
        self.frames_written += 1

    def _finish(self) -> str:
        """Close the pipe, wait for ffmpeg, return its stderr."""
        process = self._process
        if process is None:
            return ""
        # Cleared first so a second call (write's error path, then __exit__) is a harmless no-op.
        self._process = None
        try:
            # Closing stdin is ffmpeg's end-of-input signal: it then writes the trailer and exits.
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except BrokenPipeError:
            # ffmpeg already gone; its exit code below reports the real problem.
            pass
        stderr = ""
        if process.stderr is not None:
            # errors="replace": a stray non-UTF-8 byte must not hide the actual error message.
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            process.stderr.close()
        code = process.wait()
        if code != 0:
            return stderr.strip() or f"ffmpeg exited with code {code}"
        return ""

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Finish the file, or delete it if anything went wrong.

        Returning None (not True) lets the caller's own exception keep propagating. Raises
        ``VideoEncodeError`` when ffmpeg failed or no frame was written.
        """
        # Always finish first, so ffmpeg is never left running whatever happens next.
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
    # frozen=True: a probe result is a fact about a file and should not be edited after the fact.
    has_video: bool
    has_audio: bool
    width: int
    height: int
    fps: float
    frame_count: int          # 0 when the container does not record it
    video_duration: float
    audio_duration: float
    video_codec: str
    audio_codec: str

    @property
    def duration_gap(self) -> float:
        """Absolute difference between the two stream durations, in seconds."""
        # A file with one stream cannot drift out of sync with itself.
        if not (self.has_video and self.has_audio):
            return 0.0
        return abs(self.video_duration - self.audio_duration)

    def to_dict(self) -> Dict[str, object]:
        """The camelCase JSON shape used in API responses, rounded to milliseconds."""
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
    """
    Turn ffprobe's fractional rate (``"30000/1001"``) into a float (29.97); 0.0 if unreadable.

    ffprobe reports rates as fractions because NTSC rates such as 29.97 are not exact decimals.
    """
    try:
        numerator, _, denominator = value.partition("/")
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _float(value: object) -> float:
    """``float(value)``, or 0.0 when ffprobe left the field out or wrote something non-numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def parse_probe(path: Union[str, Path], payload: Dict[str, object]) -> MediaInfo:
    """Build a ``MediaInfo`` from ffprobe's JSON (split out for testing)."""
    streams = payload.get("streams") or []
    # The container's duration is the fallback for streams that do not report their own (WebM, some
    # MP3s); 0.0 means neither did.
    container = _float((payload.get("format") or {}).get("duration"))  # type: ignore[union-attr]
    # next(generator, None): the first matching stream, or None. Only the first of each kind is used.
    video = next((s for s in streams if s.get("codec_type") == "video"), None)  # type: ignore[union-attr]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)  # type: ignore[union-attr]
    # nb_frames is missing or "N/A" for some containers; that is not an error, it is just unknown.
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
    # Checked here so a missing file gives a clear FileNotFoundError, not ffprobe's own wording.
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path}")
    result = subprocess.run(
        [
            require_binary("ffprobe"),
            "-v", "error",  # quiet unless something is wrong
            "-show_streams",
            "-show_format",
            "-of", "json",  # machine-readable output instead of ffprobe's text report
            str(path),
        ],
        capture_output=True,
        text=True,  # decode stdout/stderr to str, since json.loads wants text
        check=False,  # inspect the exit code ourselves, to raise our own typed error
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
    # This is a generator function (it uses ``yield``): calling it returns an iterator and runs no code
    # until the first frame is asked for. Frames are produced one at a time, so a long video is never
    # held in memory whole.
    # Probe first: the frame size is needed to know how many bytes make one frame.
    info = probe(path)
    if not info.has_video:
        raise VideoEncodeError(f"{Path(path).name} has no video stream")
    cmd = [require_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if fps:
        # -r after -i applies to the output: ffmpeg drops or duplicates frames to hit this rate.
        cmd += ["-r", str(fps)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = info.width * info.height * 3
    # stderr is discarded: a decode error simply ends the stream early, which the caller sees as fewer
    # frames.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert process.stdout is not None  # narrows the Optional type for the type checker
        while True:
            chunk = process.stdout.read(frame_bytes)
            # A short read means end of stream; a partial frame cannot be shown, so it is dropped.
            if len(chunk) < frame_bytes:
                break
            # frombuffer wraps the bytes without copying, so the frame is read-only; copy it to edit.
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(info.height, info.width, 3)
    finally:
        # finally also runs when the caller stops iterating early (e.g. ``break`` after N frames):
        # closing stdout makes ffmpeg exit, and wait() reaps it so no zombie process is left.
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
            "-vn",  # ignore any video stream
            "-ac", "1",  # mix down to one channel (mono)
            "-ar", str(sample_rate),  # resample to the rate the caller's model expects
            "-f", "s16le",  # raw signed 16-bit little-endian samples, no WAV header
            "-",  # to stdout
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoEncodeError(
            f"ffmpeg could not decode audio from {Path(path).name}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    # int16 runs from -32768 to 32767; dividing by 32768 maps it into [-1, 1), the float convention
    # every audio model here uses.
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
