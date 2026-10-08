"""
Live interactive avatar: text in, audio chunks and animated frames out (R-18, R-19).

A file render waits for the whole speech before it draws anything. A live
session cannot: it speaks sentence by sentence and sends each sentence as soon
as it exists. For every sentence the session

    1. synthesises the speech and measures its phoneme timing (the router),
    2. sends the audio as one binary chunk,
    3. renders the avatar's frames for that audio and sends each as a JPEG,

while the *next* sentence is already being synthesised on another thread, so
the client is never starved at a sentence boundary as long as synthesis runs
faster than speech.

Wire format. Control messages are JSON text. Media is binary, one WebSocket
message per item, with a 13-byte big-endian header so the client needs no
parsing beyond a ``DataView``:

    kind(u8)  chunk(u32)  index(u32)  presentation_ms(u32)  payload...

``kind`` 1 is PCM16 mono audio (``index`` 0), ``kind`` 2 is one JPEG frame
(``index`` is its number within the chunk). ``presentation_ms`` is when the
item is due on the session's timeline, so the client can play audio and draw
frames in step without trusting arrival times.

Nothing here knows about sockets: ``stream_text`` is an async generator of
events, which keeps it testable and keeps the transport in ``app.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np

import audit_log
from avatar_store import AvatarConsentError, AvatarStore
from face_animation import AnimationTrack, build_animation, energy_envelope, seed_from_job_id, speech_energy_envelope
from face_engine import FACE_ENGINE_LOCK, shared_face_engine
from face_warp import PortraitAnimator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = PROJECT_ROOT / "outputs" / "live"

KIND_AUDIO = 1
KIND_FRAME = 2
HEADER = struct.Struct("!BIII")

JPEG_QUALITY = 80
# A sentence longer than this is split at its last space before the limit, so a
# run-on paragraph does not become one long synthesis before any audio is sent.
MAX_CHUNK_CHARS = 220

# One synthesis at a time across every session (the router serialises anyway;
# this keeps the queue visible and bounded). Rendering is numpy/OpenCV and may
# run two at a time.
# Streaming *audio* input (R-41): the client sends its own speech as PCM16 chunks and the face
# follows it. There is no transcript, so there are no phonemes to align: the mouth opens and
# closes with the loudness of the audio. That is an estimate, and every message about it says so.
AUDIO_DRIVE = "audio-energy"
AUDIO_DRIVE_NOTE = "mouth estimated from the audio's loudness; not phoneme-aligned"
MAX_PCM_SECONDS = 2.0       # one binary message carries at most this much audio
# The quietest level treated as speech when nothing louder has been heard yet (about -34 dBFS);
# below it a stream of room noise keeps the mouth shut instead of being normalised up to "talking".
MIN_AUDIO_REFERENCE = 0.02
# Steps quieter than this (RMS ~ -40 dBFS) count as silence whatever the reference: a microphone's
# room noise sits around -50 to -60 dBFS and would otherwise flicker the mouth open.
NOISE_GATE = 0.01

_SYNTH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-synth")
_RENDER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-render")


class LiveError(RuntimeError):
    """A live session cannot do what was asked; the message says how to fix it."""


def pack_media(kind: int, chunk: int, index: int, presentation_ms: int, payload: bytes) -> bytes:
    return HEADER.pack(kind, chunk, index, max(0, int(presentation_ms))) + payload


def unpack_media(message: bytes) -> tuple[int, int, int, int, bytes]:
    kind, chunk, index, presentation_ms = HEADER.unpack_from(message)
    return kind, chunk, index, presentation_ms, message[HEADER.size:]


_SENTENCE_END = re.compile(r"(?<=[.!?…।。！？])\s+")


def split_sentences(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split speech into chunks that can each be sent as soon as they are ready.

    Splits after sentence punctuation (including the Devanagari danda and CJK
    stops), then cuts any piece still longer than ``max_chars`` at a space.
    Pieces with no letter or digit in them ("...", a stray bullet) are dropped:
    there is nothing to say.
    """
    pieces: List[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        pieces.append(sentence)
    return [p for p in pieces if any(ch.isalnum() for ch in p)]


@dataclass
class ChunkAudio:
    """One synthesised sentence: audio, its measured timing, and what made it."""

    index: int
    text: str
    pcm16: bytes
    sample_rate: int
    duration_seconds: float
    timestamps: List[Dict[str, Any]]
    alignment_method: Optional[str]
    model: str
    emotion_vector: Dict[str, float]
    energy: np.ndarray
    synth_ms: float


@dataclass
class LiveEvent:
    """What ``stream_text`` yields. ``kind`` is audio | frame | chunk | done."""

    kind: str
    payload: bytes = b""
    meta: Dict[str, Any] = field(default_factory=dict)


class LiveSession:
    def __init__(
        self,
        router: Any,
        store: AvatarStore,
        avatar_id: str,
        language: str = "en",
        mode: str = "fast",
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        fps: int = 25,
        max_side: int = 384,
        speaker_wav: Optional[str] = None,
        clone_engine: Optional[str] = None,
    ) -> None:
        self.router = router
        self.store = store
        self.avatar_id = avatar_id
        self.language = language
        self.mode = mode
        self.emotion = emotion
        self.emotion_intensity = emotion_intensity
        self.fps = fps
        self.max_side = max_side
        self.speaker_wav = speaker_wav
        self.clone_engine = clone_engine
        self.session_id = uuid.uuid4().hex[:12]
        self.animator: Optional[PortraitAnimator] = None
        self.width = self.height = 0
        # Where the next chunk starts on the session's timeline (ms).
        self.timeline_ms = 0.0
        # Loudest level of the client's own audio so far (streaming audio input).
        self.audio_reference = 0.0
        LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # -- blocking work; each runs on a worker thread -----------------------

    def open(self) -> Dict[str, Any]:
        """Check consent, load the photo, find the face and build the animator."""
        import cv2

        import video_io

        try:
            image = self.store.load_image(self.avatar_id)  # raises if consent is missing
        except AvatarConsentError as err:
            audit_log.shared_audit().record("face_refused", subject=self.avatar_id, basis=None, reason=str(err), live=True)
            raise
        provenance_record = self.store.get(self.avatar_id).provenance
        audit_log.shared_audit().record(
            "face_use", subject=self.avatar_id, basis=str(provenance_record.get("consentBasis") or provenance_record.get("source") or ""),
            source=provenance_record.get("source"), live=True, session=self.session_id,
        )
        width, height = video_io.fit_within(image.shape[1], image.shape[0], self.max_side, self.max_side)
        if (width, height) != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        with FACE_ENGINE_LOCK:
            analysis = shared_face_engine().analyze(image)
        self.animator = PortraitAnimator(image, analysis)
        self.width, self.height = width, height
        return {"width": width, "height": height, "fps": self.fps}

    def synth_chunk(self, index: int, text: str) -> ChunkAudio:
        """Speak one sentence and measure when each phoneme happens."""
        import soundfile as sf

        started = time.perf_counter()
        name = f"live/{self.session_id}-{index}.wav"
        result = self.router.synthesize(
            text,
            mode=self.mode,
            language=self.language,
            speaker_wav=self.speaker_wav,
            output_filename=name,
            return_alignment=True,
            emotion=self.emotion,
            emotion_intensity=self.emotion_intensity,
            clone_engine=self.clone_engine,
        )
        path = Path(result.output_path)
        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            energy = speech_energy_envelope(path)
        finally:
            path.unlink(missing_ok=True)
        if not result.phoneme_timestamps:
            raise LiveError(
                "the speech came back with no phoneme timing, so there is nothing to move the mouth; "
                "the forced aligner could not run"
            )
        from emotion_engine import to_render_emotion_vector

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return ChunkAudio(
            index=index,
            text=text,
            pcm16=pcm,
            sample_rate=int(rate),
            duration_seconds=len(audio) / float(rate),
            timestamps=list(result.phoneme_timestamps),
            alignment_method=result.alignment_method,
            model=result.model,
            emotion_vector=to_render_emotion_vector((result.emotion or {}).get("vector") or {}),
            energy=energy,
            synth_ms=(time.perf_counter() - started) * 1000.0,
        )

    def build_track(self, chunk: ChunkAudio) -> AnimationTrack:
        return build_animation(
            chunk.timestamps,
            duration_seconds=chunk.duration_seconds,
            fps=self.fps,
            emotion_vector=chunk.emotion_vector,
            seed=seed_from_job_id(f"{self.session_id}-{chunk.index}"),
            energy_envelope=chunk.energy,
        )

    def render_jpeg(self, track: AnimationTrack, index: int) -> bytes:
        import cv2

        assert self.animator is not None
        frame = self.animator.render(track.frame(index))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise LiveError("a frame could not be encoded as JPEG")
        return encoded.tobytes()

    def audio_track(self, index: int, pcm16: bytes, sample_rate: int) -> AnimationTrack:
        """Frames for one chunk of the client's own audio: the mouth follows its loudness (``AUDIO_DRIVE``)."""
        mono = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        duration = len(mono) / float(sample_rate)
        rms, _ = energy_envelope(mono, sample_rate, reference=1.0)  # raw RMS per grid step (full scale = 1)
        speech = rms[rms > 1e-4]
        loudest = float(np.percentile(speech, 95)) if speech.size else 0.0
        # Normalise by the loudest level heard so far (this chunk included), never below the floor:
        # a quiet chunk after a loud one is drawn as quiet, and room noise alone keeps the mouth shut.
        self.audio_reference = max(MIN_AUDIO_REFERENCE, self.audio_reference, loudest)
        envelope = np.clip(rms / self.audio_reference, 0.0, 1.0).astype(np.float32)
        envelope[rms < NOISE_GATE] = 0.0
        # One open-vowel shape held for the whole chunk; the energy gate in build_animation opens and
        # shuts it with the loudness. Without phonemes there is nothing better to pick.
        held = [{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": duration * 1000.0}]
        emotion = None
        if self.emotion:
            from emotion_engine import to_render_emotion_vector

            emotion = to_render_emotion_vector({self.emotion: self.emotion_intensity})
        return build_animation(held, duration_seconds=duration, fps=self.fps, emotion_vector=emotion,
                               seed=seed_from_job_id(f"{self.session_id}-a{index}"), energy_envelope=envelope)

    def cleanup(self) -> None:
        for leftover in LIVE_DIR.glob(f"{self.session_id}-*.wav"):
            leftover.unlink(missing_ok=True)


async def stream_text(session: LiveSession, text: str, with_frames: bool = True) -> AsyncGenerator[LiveEvent, None]:
    """
    Speak ``text`` and yield, in order: for each sentence an ``audio`` event, its
    ``frame`` events, then a ``chunk`` event with the timings; finally ``done``.

    Closing the generator early (client disconnected, interrupt) stops it; the
    sentence being synthesised at that moment finishes on its worker thread and
    its audio file is removed, but nothing more is started.
    """
    loop = asyncio.get_running_loop()
    sentences = split_sentences(text)
    if not sentences:
        raise LiveError("there is nothing to say: the text has no words in it")

    t_say = time.perf_counter()
    first_audio_ms: Optional[float] = None
    first_frame_ms: Optional[float] = None
    first_gap_ms: Optional[float] = None
    pending: Optional["asyncio.Future[ChunkAudio]"] = loop.run_in_executor(
        _SYNTH_POOL, session.synth_chunk, 0, sentences[0]
    )
    try:
        for k in range(len(sentences)):
            assert pending is not None
            chunk: ChunkAudio = await pending
            pending = (
                loop.run_in_executor(_SYNTH_POOL, session.synth_chunk, k + 1, sentences[k + 1])
                if k + 1 < len(sentences)
                else None
            )
            start_ms = session.timeline_ms
            yield LiveEvent("audio", pack_media(KIND_AUDIO, k, 0, int(start_ms), chunk.pcm16), {
                "chunk": k, "text": chunk.text, "sampleRate": chunk.sample_rate,
                "durationMs": round(chunk.duration_seconds * 1000, 1), "startMs": round(start_ms, 1),
            })
            t_audio = time.perf_counter()
            if first_audio_ms is None:
                first_audio_ms = (t_audio - t_say) * 1000.0

            frames = 0
            first_gap: Optional[float] = None
            render_ms = 0.0
            if with_frames:
                track = await loop.run_in_executor(_RENDER_POOL, session.build_track, chunk)
                for i in range(track.frame_count):
                    t0 = time.perf_counter()
                    jpeg = await loop.run_in_executor(_RENDER_POOL, session.render_jpeg, track, i)
                    render_ms += (time.perf_counter() - t0) * 1000.0
                    if first_gap is None:
                        first_gap = (time.perf_counter() - t_audio) * 1000.0
                    if first_frame_ms is None:
                        first_frame_ms = (time.perf_counter() - t_say) * 1000.0
                        first_gap_ms = first_gap
                    frames += 1
                    yield LiveEvent("frame", pack_media(KIND_FRAME, k, i, int(start_ms + i * 1000.0 / session.fps), jpeg))
            session.timeline_ms += chunk.duration_seconds * 1000.0
            yield LiveEvent("chunk", meta={
                "chunk": k, "text": chunk.text, "model": chunk.model, "alignmentMethod": chunk.alignment_method,
                "audioMs": round(chunk.duration_seconds * 1000, 1), "synthMs": round(chunk.synth_ms, 1),
                "frames": frames, "renderMsPerFrame": round(render_ms / frames, 1) if frames else None,
                "firstFrameAfterAudioMs": round(first_gap, 1) if first_gap is not None else None,
            })
        yield LiveEvent("done", meta={
            "chunks": len(sentences), "textToFirstAudioMs": round(first_audio_ms or 0, 1),
            "textToFirstFrameMs": round(first_frame_ms, 1) if first_frame_ms is not None else None,
            "firstFrameAfterFirstAudioMs": round(first_gap_ms, 1) if first_gap_ms is not None else None,
            "totalMs": round((time.perf_counter() - t_say) * 1000, 1),
        })
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


def check_pcm(data: bytes, sample_rate: int) -> None:
    """Refuse a chunk the server cannot animate, with the reason (raises ``LiveError``)."""
    if not data:
        raise LiveError("an audio chunk was empty")
    if len(data) % 2:
        raise LiveError("audio chunks must be 16-bit little-endian mono PCM (an even number of bytes)")
    limit = int(MAX_PCM_SECONDS * sample_rate) * 2
    if len(data) > limit:
        raise LiveError(f"an audio chunk may hold at most {MAX_PCM_SECONDS:g} s ({limit} bytes at {sample_rate} Hz); split it")


async def stream_audio_chunk(session: LiveSession, index: int, pcm16: bytes, sample_rate: int) -> AsyncGenerator[LiveEvent, None]:
    """
    Animate one chunk of the client's own speech: yield its ``frame`` events, then a ``chunk``
    event. No audio is sent back; the client already has it. The frames sit on the same timeline
    as text speech, so a session can mix the two.
    """
    loop = asyncio.get_running_loop()
    check_pcm(pcm16, sample_rate)
    started = time.perf_counter()
    track = await loop.run_in_executor(_RENDER_POOL, session.audio_track, index, pcm16, sample_rate)
    start_ms = session.timeline_ms
    audio_ms = len(pcm16) / 2 / sample_rate * 1000.0
    # Frames sit on the session's one global grid (every 1000/fps ms) rather than restarting at each
    # chunk: a 0.5 s chunk is 12.5 frames, and rounding each chunk up would drift to 26 fps and
    # put frames off the grid. Each grid slot inside this chunk takes the nearest frame of its track.
    frame_ms = 1000.0 / session.fps
    first, end = math.ceil(start_ms / frame_ms - 1e-9), math.ceil((start_ms + audio_ms) / frame_ms - 1e-9)
    frames = 0
    for slot in range(first, end):
        local = min(track.frame_count - 1, int((slot * frame_ms - start_ms) / frame_ms))
        jpeg = await loop.run_in_executor(_RENDER_POOL, session.render_jpeg, track, local)
        yield LiveEvent("frame", pack_media(KIND_FRAME, index, frames, int(round(slot * frame_ms)), jpeg))
        frames += 1
    session.timeline_ms += audio_ms
    yield LiveEvent("chunk", meta={
        "chunk": index, "drive": AUDIO_DRIVE, "note": AUDIO_DRIVE_NOTE, "audioMs": round(audio_ms, 1),
        "frames": frames, "startMs": round(start_ms, 1),
        "processingMs": round((time.perf_counter() - started) * 1000.0, 1),
    })
