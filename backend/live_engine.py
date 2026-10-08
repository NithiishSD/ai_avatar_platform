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

Where it sits: ``app.py``'s live WebSocket builds a :class:`LiveSession`, calls ``open`` once, then
forwards every event from :func:`stream_text` (typed text) or :func:`stream_audio_chunk` (the user's
own microphone audio) to the browser. Speech comes from the voice router, face motion from
``face_animation``, and pixels from ``face_warp.PortraitAnimator``.

Concepts used here:
  * *asyncio and blocking work*. The server runs one event loop that switches between connections
    whenever a coroutine ``await``s. Synthesis and rendering are slow CPU work that never awaits, so
    running them on the loop would freeze every other connection. ``loop.run_in_executor(pool, fn,
    *args)`` runs ``fn`` on a thread from ``pool`` and returns a future the loop can ``await``.
  * *Async generators*. An ``async def`` that ``yield``s produces values one at a time to an
    ``async for`` loop and can ``await`` between them. The consumer pulls the next event when it is
    ready to send it, so events never pile up in memory.
  * *Pipelining*. Sentence k+1 is submitted for synthesis before sentence k's frames are rendered, so
    the two overlap in time instead of running back to back.
  * *PCM16*. Raw audio as signed 16-bit integers, one per sample; -32768..32767 maps to -1.0..1.0.
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
# Per-sentence WAV files land here briefly; each is deleted as soon as it has been read.
LIVE_DIR = PROJECT_ROOT / "outputs" / "live"

# The ``kind`` byte of the binary header (see the module docstring).
KIND_AUDIO = 1
KIND_FRAME = 2
# struct format: "!" network (big-endian) byte order, "B" one unsigned byte, "I" unsigned 32-bit int.
# 1 + 4 + 4 + 4 = 13 bytes. A compiled Struct is reused for every message instead of re-parsing it.
HEADER = struct.Struct("!BIII")

# OpenCV's JPEG quality scale is 0..100; 80 is the setting this module sends frames at.
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

# The two worker pools named in the note above "Streaming audio input": one synthesis thread shared
# by every session, two render threads. Named threads make them easy to spot in a stack dump.
_SYNTH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-synth")
_RENDER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-render")


class LiveError(RuntimeError):
    """A live session cannot do what was asked; the message says how to fix it."""


def pack_media(kind: int, chunk: int, index: int, presentation_ms: int, payload: bytes) -> bytes:
    """One binary WebSocket message: the 13-byte header followed by the payload bytes.

    ``presentation_ms`` is clamped at 0 because the header field is unsigned.
    """
    return HEADER.pack(kind, chunk, index, max(0, int(presentation_ms))) + payload


def unpack_media(message: bytes) -> tuple[int, int, int, int, bytes]:
    """The inverse of :func:`pack_media`: ``(kind, chunk, index, presentation_ms, payload)``."""
    kind, chunk, index, presentation_ms = HEADER.unpack_from(message)
    return kind, chunk, index, presentation_ms, message[HEADER.size:]


# "(?<=...)" is a lookbehind: split on the whitespace *after* a sentence mark, so the mark stays
# attached to its sentence instead of being consumed by the split.
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
            # The last space before the limit; with no space at all (one huge word), cut hard at the limit.
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        pieces.append(sentence)
    return [p for p in pieces if any(ch.isalnum() for ch in p)]


@dataclass
class ChunkAudio:
    """One synthesised sentence: audio, its measured timing, and what made it.

    ``index`` is the sentence's position in the text; ``pcm16`` the audio as sent on the wire;
    ``timestamps`` the phoneme timing from the forced aligner, with ``alignment_method`` naming how
    it was found; ``energy`` the loudness envelope used to gate the mouth; ``synth_ms`` how long
    synthesis took, reported in the chunk event.
    """

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
    """What ``stream_text`` yields. ``kind`` is audio | frame | chunk | done.

    Media events (audio, frame) carry ``payload`` bytes to send as a binary message; the others
    carry ``meta``, which the transport sends as JSON text.
    """

    kind: str
    payload: bytes = b""
    meta: Dict[str, Any] = field(default_factory=dict)


class LiveSession:
    """One viewer's live avatar: the chosen face and voice settings, and the shared timeline.

    The methods in the "blocking work" section are plain functions run on worker threads by the
    async streamers below; the session itself holds no sockets.
    """

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
        voice: Optional[str] = None,
    ) -> None:
        self.router = router
        self.voice = voice
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
        # 12 hex characters: short enough for file names, random enough not to collide between sessions.
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
        # Live frames are kept small (max_side, 384 px by default) so each one renders and encodes fast.
        width, height = video_io.fit_within(image.shape[1], image.shape[0], self.max_side, self.max_side)
        if (width, height) != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # The face is found once per session; every frame after that reuses this analysis.
        with FACE_ENGINE_LOCK:
            analysis = shared_face_engine().analyze(image)
        self.animator = PortraitAnimator(image, analysis)
        self.width, self.height = width, height
        return {"width": width, "height": height, "fps": self.fps}

    def synth_chunk(self, index: int, text: str) -> ChunkAudio:
        """Speak one sentence and measure when each phoneme happens."""
        import soundfile as sf

        started = time.perf_counter()
        # A unique name per sentence, so two sessions or two sentences never overwrite each other's file.
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
            voice=self.voice,
        )
        path = Path(result.output_path)
        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
            # Stereo is averaged to mono: the wire format carries one channel.
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            energy = speech_energy_envelope(path)
        finally:
            # Deleted even if reading failed: the audio is sent over the socket, not kept on disk.
            path.unlink(missing_ok=True)
        # No silent fallback (golden rule 1): without phoneme timing the mouth cannot follow the speech.
        if not result.phoneme_timestamps:
            raise LiveError(
                "the speech came back with no phoneme timing, so there is nothing to move the mouth; "
                "the forced aligner could not run"
            )
        from emotion_engine import to_render_emotion_vector

        # float -1..1 to PCM16. Clipping first stops an over-range sample wrapping round to the other
        # extreme. "<i2" is little-endian signed 16-bit, which the browser's decoder expects.
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
        """The per-frame face motion for one sentence, from its phoneme timing and loudness.

        The seed is derived from the session and chunk, so idle motion (blinks, sway) is repeatable.
        """
        return build_animation(
            chunk.timestamps,
            duration_seconds=chunk.duration_seconds,
            fps=self.fps,
            emotion_vector=chunk.emotion_vector,
            seed=seed_from_job_id(f"{self.session_id}-{chunk.index}"),
            energy_envelope=chunk.energy,
        )

    def render_jpeg(self, track: AnimationTrack, index: int) -> bytes:
        """Draw frame ``index`` of ``track`` and return it JPEG-encoded; raises LiveError on failure."""
        import cv2

        # open() must have run first; the assert also narrows the Optional type for the type checker.
        assert self.animator is not None
        frame = self.animator.render(track.frame(index))
        # OpenCV encodes BGR, so the RGB frame is converted first or the colours come out swapped.
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise LiveError("a frame could not be encoded as JPEG")
        return encoded.tobytes()

    def audio_track(self, index: int, pcm16: bytes, sample_rate: int) -> AnimationTrack:
        """Frames for one chunk of the client's own audio: the mouth follows its loudness (``AUDIO_DRIVE``)."""
        # PCM16 back to float -1..1 (32768 = 2**15, the size of the negative range).
        mono = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        duration = len(mono) / float(sample_rate)
        rms, _ = energy_envelope(mono, sample_rate, reference=1.0)  # raw RMS per grid step (full scale = 1)
        # Ignore near-digital-silence steps, then take the 95th percentile rather than the max, so one
        # click or pop does not set the scale for everything after it.
        speech = rms[rms > 1e-4]
        loudest = float(np.percentile(speech, 95)) if speech.size else 0.0
        # Normalise by the loudest level heard so far (this chunk included), never below the floor:
        # a quiet chunk after a loud one is drawn as quiet, and room noise alone keeps the mouth shut.
        self.audio_reference = max(MIN_AUDIO_REFERENCE, self.audio_reference, loudest)
        envelope = np.clip(rms / self.audio_reference, 0.0, 1.0).astype(np.float32)
        # Boolean-mask assignment: every step whose RMS is under the gate is set to 0 (mouth closed).
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
        """Delete any of this session's WAV files that were left behind (e.g. a cancelled synthesis)."""
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
    # The loop this coroutine runs on; executor futures are bound to it.
    loop = asyncio.get_running_loop()
    sentences = split_sentences(text)
    if not sentences:
        raise LiveError("there is nothing to say: the text has no words in it")

    # perf_counter is a monotonic clock for measuring intervals; it never jumps if the wall clock changes.
    # These timings are reported in the "chunk" and "done" events so latency is measured, not guessed.
    t_say = time.perf_counter()
    first_audio_ms: Optional[float] = None
    first_frame_ms: Optional[float] = None
    first_gap_ms: Optional[float] = None
    # Start the first sentence before the loop; each pass then starts the next one (pipelining).
    pending: Optional["asyncio.Future[ChunkAudio]"] = loop.run_in_executor(
        _SYNTH_POOL, session.synth_chunk, 0, sentences[0]
    )
    try:
        for k in range(len(sentences)):
            assert pending is not None
            chunk: ChunkAudio = await pending
            # Queue sentence k+1 now, so it synthesises while sentence k is being rendered and sent.
            pending = (
                loop.run_in_executor(_SYNTH_POOL, session.synth_chunk, k + 1, sentences[k + 1])
                if k + 1 < len(sentences)
                else None
            )
            # Audio is yielded before any frame, so the client can start playback at once.
            start_ms = session.timeline_ms
            yield LiveEvent("audio", pack_media(KIND_AUDIO, k, 0, int(start_ms), chunk.pcm16), {
                "chunk": k, "text": chunk.text, "sampleRate": chunk.sample_rate,
                "durationMs": round(chunk.duration_seconds * 1000, 1), "startMs": round(start_ms, 1),
            })
            t_audio = time.perf_counter()
            # The first sentence's numbers are the latency a user feels: time to first sound and first frame.
            if first_audio_ms is None:
                first_audio_ms = (t_audio - t_say) * 1000.0

            frames = 0
            first_gap: Optional[float] = None
            render_ms = 0.0
            if with_frames:
                track = await loop.run_in_executor(_RENDER_POOL, session.build_track, chunk)
                # One executor call per frame, so each JPEG is sent as soon as it exists.
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
                    # Frame i of this chunk is due i / fps seconds after the chunk's audio starts.
                    yield LiveEvent("frame", pack_media(KIND_FRAME, k, i, int(start_ms + i * 1000.0 / session.fps), jpeg))
            # The next sentence starts where this one's audio ends.
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
        # Runs when the consumer closes the generator early too. cancel() only stops a job that has
        # not started; one already running finishes on its thread, as the docstring says.
        if pending is not None and not pending.done():
            pending.cancel()


def check_pcm(data: bytes, sample_rate: int) -> None:
    """Refuse a chunk the server cannot animate, with the reason (raises ``LiveError``)."""
    if not data:
        raise LiveError("an audio chunk was empty")
    # Two bytes per 16-bit sample, so an odd length cannot be whole samples.
    if len(data) % 2:
        raise LiveError("audio chunks must be 16-bit little-endian mono PCM (an even number of bytes)")
    # Bound the work one message can ask for; the limit is in bytes (samples x 2).
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
    # Validate before any work is queued, so a bad chunk costs nothing.
    check_pcm(pcm16, sample_rate)
    started = time.perf_counter()
    track = await loop.run_in_executor(_RENDER_POOL, session.audio_track, index, pcm16, sample_rate)
    start_ms = session.timeline_ms
    # bytes / 2 = samples; samples / rate = seconds.
    audio_ms = len(pcm16) / 2 / sample_rate * 1000.0
    # Frames sit on the session's one global grid (every 1000/fps ms) rather than restarting at each
    # chunk: a 0.5 s chunk is 12.5 frames, and rounding each chunk up would drift to 26 fps and
    # put frames off the grid. Each grid slot inside this chunk takes the nearest frame of its track.
    frame_ms = 1000.0 / session.fps
    # The first and one-past-last grid slots inside [start_ms, start_ms + audio_ms). The tiny 1e-9 stops
    # floating-point error (e.g. 2.0000000001) from pushing an exact slot to the next one.
    first, end = math.ceil(start_ms / frame_ms - 1e-9), math.ceil((start_ms + audio_ms) / frame_ms - 1e-9)
    frames = 0
    for slot in range(first, end):
        # Which frame of this chunk's own track falls at this grid slot; min() guards the last slot.
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
