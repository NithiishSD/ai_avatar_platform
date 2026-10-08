"""
audio_utils.py — Audio validation, format conversion, and voice sample discovery.

Responsibilities:
  - Scan inputs/ directory and list all valid audio files for voice cloning.
  - Validate audio files against XTTS-v2 requirements (duration, channels, format).
  - Convert any supported format (MP3, FLAC, OGG, M4A, etc.) to WAV 24 kHz mono
    so XTTS-v2 never receives a mismatched file.

Supported source formats (via librosa / soundfile / ffmpeg):
  WAV, FLAC, OGG, MP3, M4A, AAC, MP4, WEBM, AIFF, AU

XTTS-v2 requirements:
  - Format   : WAV (PCM 16-bit or float32)
  - Channels : mono (1) — stereo is down-mixed
  - Rate     : 22 050 Hz or 24 000 Hz (we always output 24 000)
  - Duration : at least 3 s; 30–60 s recommended for best cloning quality

Concepts used below:
  * **Sample rate** is how many amplitude values describe one second of sound.
    A model trained at 24 kHz expects exactly that many per second; feed it
    another rate and speech plays at the wrong pitch and speed.
  * **Resampling** recomputes the waveform at a new rate (librosa filters and
    interpolates), so 44.1 kHz audio becomes 24 kHz without changing pitch.
  * **Down-mixing** averages the channels of a stereo file into one mono track.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XTTS_OUTPUT_SAMPLE_RATE: int = 24_000      # Hz — XTTS-v2 native rate
XTTS_MIN_DURATION_SECONDS: float = 3.0    # hard minimum — will raise below this
XTTS_WARN_MIN_DURATION: float = 10.0      # warn if shorter than this
XTTS_IDEAL_MIN_DURATION: float = 30.0     # ideal for good speaker similarity
XTTS_MAX_DURATION_SECONDS: float = 300.0  # 5 min — reject obviously wrong files

# A set so the per-file extension check is a constant-time lookup. Matching is
# done on the lower-cased suffix, so ".WAV" and ".wav" are treated the same.
SUPPORTED_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".mp3",
    ".m4a", ".aac", ".mp4", ".webm", ".aif", ".aiff", ".au",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AudioInfo:
    """Metadata about an audio file in inputs/."""
    filename: str          # basename, e.g. "voice_sample.wav"
    path: str              # absolute path
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str            # file extension without dot, e.g. "wav"
    size_bytes: int

    @property
    def ready_for_cloning(self) -> bool:
        """True if file is already WAV 24 kHz mono (no conversion needed)."""
        return (
            self.format == "wav"
            and self.sample_rate == XTTS_OUTPUT_SAMPLE_RATE
            and self.channels == 1
            and self.duration_seconds >= XTTS_MIN_DURATION_SECONDS
        )

    @property
    def duration_label(self) -> str:
        """Human-readable length such as ``"1m 5s"``, or ``"42s"`` under a minute."""
        mins = int(self.duration_seconds // 60)
        secs = int(self.duration_seconds % 60)
        return f"{mins}m {secs}s" if mins else f"{secs}s"


class AudioValidationError(ValueError):
    """Raised when an audio file fails validation and cannot be used for cloning."""


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def list_voice_samples(input_dir: Path) -> List[AudioInfo]:
    """
    Scan *input_dir* for all supported audio files and return their metadata.

    Returns an empty list if the directory is empty or does not exist.
    Files that cannot be probed (corrupted, unknown codec) are skipped with a warning.
    """
    if not input_dir.exists():
        logger.warning("inputs/ directory does not exist: %s", input_dir)
        return []

    results: List[AudioInfo] = []
    # sorted() gives a stable order, so the UI list does not reshuffle between calls.
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            info = probe_audio(path)
            results.append(info)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the rest
            logger.warning("Skipping %s — could not probe: %s", path.name, exc)

    return results


def probe_audio(path: Path) -> AudioInfo:
    """
    Return AudioInfo for *path* without loading the full audio into memory.
    Uses soundfile for WAV/FLAC and librosa for everything else.
    """
    suffix = path.suffix.lower().lstrip(".")
    try:
        # Fast path: soundfile can probe WAV/FLAC/OGG/AIFF without full decode
        file_info = sf.info(str(path))
        return AudioInfo(
            filename=path.name,
            path=str(path),
            duration_seconds=file_info.duration,
            sample_rate=file_info.samplerate,
            channels=file_info.channels,
            format=suffix,
            size_bytes=path.stat().st_size,
        )
    except Exception as exc:  # noqa: BLE001 - any soundfile failure means "try librosa"
        # Not an error: soundfile cannot read MP3/M4A, and librosa below can.
        logger.debug("soundfile could not probe %s (%s); trying librosa", path.name, exc)

    # Fallback: use librosa to probe (handles MP3, M4A, etc.)
    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        # librosa doesn't expose channels easily for non-WAV — default to unknown
        return AudioInfo(
            filename=path.name,
            path=str(path),
            duration_seconds=duration,
            sample_rate=0,    # unknown without full load
            channels=0,       # unknown without full load
            format=suffix,
            size_bytes=path.stat().st_size,
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot probe '{path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Validation + conversion
# ---------------------------------------------------------------------------

def validate_and_convert_for_cloning(
    input_path: Path,
    converted_dir: Optional[Path] = None,
) -> Path:
    """
    Validate *input_path* for XTTS-v2 cloning and convert if needed.

    Returns the path to a WAV 24 kHz mono file that is ready for XTTS-v2.
    If no conversion is needed, returns *input_path* unchanged.
    If conversion is needed, writes to *converted_dir* (defaults to input_path.parent)
    with suffix `_converted.wav`.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    AudioValidationError
        If the audio is too short, too long, or in an unrecognisable format.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"❌ Voice sample not found: '{input_path}'\n"
            "   Place a WAV/MP3/FLAC recording (30–60 s) into the inputs/ folder."
        )

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioValidationError(
            f"❌ Unsupported audio format '{suffix}' for '{input_path.name}'.\n"
            f"   Supported formats: {supported_list}\n"
            "   Convert the file to WAV or MP3 and place it in inputs/."
        )

    # ---- Load audio (universal path via librosa) -------------------------
    try:
        import librosa
        # sr=None keeps the file's own rate (resampling happens below, only if needed);
        # mono=False keeps the channels so the stereo check below can see them.
        audio, original_sr = librosa.load(str(input_path), sr=None, mono=False)
    except Exception as exc:
        raise AudioValidationError(
            f"❌ Cannot read audio file '{input_path.name}': {exc}\n"
            "   The file may be corrupted or use an unsupported codec.\n"
            "   Try re-encoding it: ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav"
        ) from exc

    # ---- Duration check --------------------------------------------------
    if audio.ndim == 1:
        n_samples = len(audio)
    else:
        n_samples = audio.shape[-1]  # (channels, samples) after librosa.load mono=False

    duration = n_samples / original_sr
    if duration < XTTS_MIN_DURATION_SECONDS:
        raise AudioValidationError(
            f"❌ Audio too short: '{input_path.name}' is {duration:.1f}s "
            f"(minimum {XTTS_MIN_DURATION_SECONDS}s required).\n"
            f"   Ideal: 30–60 s of clean speech for best speaker similarity."
        )
    if duration > XTTS_MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"❌ Audio too long: '{input_path.name}' is {duration:.0f}s "
            f"(maximum {XTTS_MAX_DURATION_SECONDS}s).\n"
            "   Trim it to 30–60 s of clean, continuous speech."
        )

    if duration < XTTS_WARN_MIN_DURATION:
        logger.warning(
            "Voice sample '%s' is %.1fs — shorter than the recommended %ds. "
            "Speaker similarity may be reduced.",
            input_path.name, duration, int(XTTS_IDEAL_MIN_DURATION),
        )

    # ---- Check if already in the right format ----------------------------
    needs_conversion = False

    if audio.ndim > 1:
        # Multi-channel — need to down-mix to mono
        needs_conversion = True
        # Averaging, not picking one channel, keeps speech that is panned to one side.
        audio = np.mean(audio, axis=0)  # shape: (samples,)
    elif audio.ndim == 1:
        pass  # already mono

    if original_sr != XTTS_OUTPUT_SAMPLE_RATE:
        needs_conversion = True
        import librosa
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=XTTS_OUTPUT_SAMPLE_RATE)

    if suffix != ".wav":
        needs_conversion = True

    if not needs_conversion:
        logger.debug("'%s' is already WAV 24 kHz mono — no conversion needed.", input_path.name)
        return input_path

    # ---- Write converted WAV ---------------------------------------------
    if converted_dir is None:
        converted_dir = input_path.parent

    converted_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    out_path = converted_dir / f"{stem}_converted.wav"

    # PCM_16 is 16-bit integer samples: the most widely readable WAV encoding.
    # soundfile scales the float32 values in [-1, 1] to that integer range.
    sf.write(str(out_path), audio.astype(np.float32), XTTS_OUTPUT_SAMPLE_RATE, subtype="PCM_16")

    logger.info(
        "Converted '%s' (%d Hz, %s) → '%s' (24000 Hz, WAV mono, %.1fs)",
        input_path.name, original_sr, suffix, out_path.name, duration,
    )
    print(
        f"  [AudioUtils] Converted '{input_path.name}' "
        f"({original_sr} Hz {suffix}) → '{out_path.name}' (24000 Hz WAV mono)"
    )

    return out_path
