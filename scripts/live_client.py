#!/usr/bin/env python3
"""
Talk to a running live avatar (WS /api/v1/live) and measure what a viewer would feel.

Opens a session, speaks the text, decodes every message the way a browser would,
and reports:

* ``textToFirstAudioMs``  - say sent -> first audio chunk received (T4.1, R-18)
* ``firstFrameAfterFirstAudioMs`` - first audio chunk -> first frame received
  (the roadmap's "< 200 ms first frame after the first chunk", N-07)
* ``textToFirstFrameMs``  - say sent -> first frame received (what the viewer waits)
* real-time viability: a client that starts playing when the first audio arrives
  needs every frame by ``first audio arrival + presentationMs``. ``lateFrames`` counts
  frames that arrive after that deadline, and ``lateMsP95`` / ``lateMsMax`` say by how
  much. Zero late frames means the stream keeps up with speech on this machine.

Every frame is decoded (size checked against the ready message) so a stream of
garbage cannot pass. ``--save DIR`` keeps a few frames and the audio.

    PYTHONPATH=backend backend/.conda/bin/python scripts/live_client.py \\
        --avatar demo --text "Hello there. This is a live avatar speaking to you."

``--audio FILE`` streams a recording instead (R-41): it is sent as 0.5 s PCM16 chunks at real-time
pace, as a microphone would, and the report adds how long each chunk took to come back as frames
and whether the mouth follows the audio: the correlation between each frame's audio loudness and
how far the lower face has moved from its quietest frame.

    ... scripts/live_client.py --avatar demo --audio inputs/ljspeech_reference.wav --seconds 10
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import struct
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("!BIII")  # must match live_engine.HEADER; kept inline so this runs standalone


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))] if ordered else 0.0


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import websockets
    from PIL import Image

    url = args.api.replace("http", "ws", 1) + "/api/v1/live"
    start = {"type": "start", "avatarId": args.avatar, "language": args.language, "mode": args.mode,
             "fps": args.fps, "maxSide": args.max_side}
    if args.emotion:
        start["emotion"] = args.emotion
    if args.speaker_wav:
        start["speakerWav"] = args.speaker_wav
    if args.clone_engine:
        start["cloneEngine"] = args.clone_engine
    if args.api_key:
        start["apiKey"] = args.api_key

    async with websockets.connect(url, max_size=None) as ws:
        opened = time.perf_counter()
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"server refused the session: {ready}")
        open_ms = (time.perf_counter() - opened) * 1000

        t_say = time.perf_counter()
        await ws.send(json.dumps({"type": "say", "text": args.text}))
        arrivals: List[tuple] = []  # (kind, chunk, index, presentation_ms, arrival_s, bytes)
        audio_meta: List[Dict[str, Any]] = []
        chunks: List[Dict[str, Any]] = []
        done: Dict[str, Any] = {}
        pcm = bytearray()
        saved = 0
        bad_frames = 0
        previous = None
        lower_face_change: List[float] = []   # mean pixel change in the lower half between consecutive frames
        while True:
            message = await ws.recv()
            now = time.perf_counter()
            if isinstance(message, bytes):
                kind, chunk, index, pres, = HEADER.unpack_from(message)[:4]
                payload = message[HEADER.size:]
                arrivals.append((kind, chunk, index, pres, now, len(payload)))
                if kind == 1:
                    pcm += payload
                elif kind == 2:
                    try:
                        image = Image.open(io.BytesIO(payload))
                        image.load()
                        if image.size != (ready["width"], ready["height"]):
                            bad_frames += 1
                        pixels = np.asarray(image.convert("L"), dtype=np.float32)
                        if previous is not None and previous.shape == pixels.shape:
                            lower_face_change.append(float(np.abs(pixels - previous)[pixels.shape[0] // 2:].mean()))
                        previous = pixels
                        if args.save and saved < 4 and index in (0, 6):
                            Path(args.save).mkdir(parents=True, exist_ok=True)
                            image.save(Path(args.save) / f"live-c{chunk}-f{index}.png")
                            saved += 1
                    except Exception:  # noqa: BLE001 - a bad frame is the finding
                        bad_frames += 1
            else:
                data = json.loads(message)
                if data["type"] == "audio":
                    audio_meta.append(data)
                elif data["type"] == "chunk":
                    chunks.append(data)
                elif data["type"] == "error":
                    raise SystemExit(f"server error: {data}")
                elif data["type"] == "done":
                    done = data
                    break
        await ws.send(json.dumps({"type": "stop"}))

    audio = [a for a in arrivals if a[0] == 1]
    frames = [a for a in arrivals if a[0] == 2]
    first_audio = audio[0][4]
    first_frame = frames[0][4] if frames else None
    # Playback starts when the first audio chunk lands; frame k is due presentation_ms later.
    late = [max(0.0, (f[4] - first_audio) * 1000 - f[3]) for f in frames]
    late_audio = [max(0.0, (a[4] - first_audio) * 1000 - a[3]) for a in audio]
    if args.save and pcm:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        with wave.open(str(Path(args.save) / "live-audio.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(ready["sampleRate"])
            out.writeframes(bytes(pcm))
    pres = [f[3] for f in frames]
    return {
        "mode": args.mode, "text": args.text, "session": {k: ready[k] for k in ("width", "height", "fps", "model", "requestId")},
        "sessionOpenMs": round(open_ms, 1),
        "chunks": len(audio), "frames": len(frames), "frameBytesMean": round(sum(f[5] for f in frames) / max(1, len(frames))),
        "audioSecondsReceived": round(len(pcm) / 2 / ready["sampleRate"], 2),
        "textToFirstAudioMs": round((first_audio - t_say) * 1000, 1),
        "firstFrameAfterFirstAudioMs": round((first_frame - first_audio) * 1000, 1) if first_frame else None,
        "textToFirstFrameMs": round((first_frame - t_say) * 1000, 1) if first_frame else None,
        "totalMs": round((arrivals[-1][4] - t_say) * 1000, 1),
        "lateFrames": sum(1 for x in late if x > 0), "lateMsP95": round(percentile(late, 95), 1), "lateMsMax": round(max(late), 1),
        "lateAudioChunks": sum(1 for x in late_audio if x > 0), "lateAudioMsMax": round(max(late_audio), 1),
        "presentationMonotonic": all(b >= a for a, b in zip(pres, pres[1:], strict=False)),
        "undecodableOrWrongSizeFrames": bad_frames,
        # Is the face actually moving? Mean pixel change of the lower half between consecutive frames;
        # a frozen portrait would read 0.0 on every frame.
        "motion": {"framesCompared": len(lower_face_change),
                   "framesIdenticalToPrevious": sum(1 for x in lower_face_change if x == 0.0),
                   "meanChange": round(sum(lower_face_change) / max(1, len(lower_face_change)), 3),
                   "maxChange": round(max(lower_face_change, default=0.0), 3)},
        "serverChunks": [{k: c.get(k) for k in ("chunk", "model", "alignmentMethod", "audioMs", "synthMs", "frames", "renderMsPerFrame", "firstFrameAfterAudioMs")} for c in chunks],
        "serverSummary": done,
    }


async def run_audio(args: argparse.Namespace) -> Dict[str, Any]:
    """Stream a recording into a live session and measure the frames that come back."""
    import numpy as np
    import soundfile as sf
    import websockets
    from PIL import Image

    data, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)[: int(args.seconds * rate)]
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(0.5 * rate) * 2  # 0.5 s of 16-bit samples
    pieces = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    url = args.api.replace("http", "ws", 1) + "/api/v1/live"
    sent_at: List[float] = []
    chunk_back: Dict[int, float] = {}
    frames: Dict[int, Any] = {}  # presentation ms -> grey lower half

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "avatarId": args.avatar, "fps": args.fps, "maxSide": args.max_side,
                                  **({"apiKey": args.api_key} if args.api_key else {})}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"server refused the session: {ready}")
        await ws.send(json.dumps({"type": "audio_start", "sampleRate": int(rate), "consentBasis": args.consent_basis}))
        audio_ready = json.loads(await ws.recv())
        if audio_ready.get("type") != "audio_ready":
            raise SystemExit(f"server refused the audio stream: {audio_ready}")

        async def send() -> None:
            began = time.perf_counter()
            for k, piece in enumerate(pieces):
                # Real-time pace: chunk k is not available before k x 0.5 s, as from a microphone.
                await asyncio.sleep(max(0.0, began + k * 0.5 - time.perf_counter()))
                sent_at.append(time.perf_counter())
                await ws.send(piece)
            await ws.send(json.dumps({"type": "audio_end"}))

        sender = asyncio.create_task(send())
        done: Dict[str, Any] = {}
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                kind, chunk, index, pres = HEADER.unpack_from(message)[:4]
                image = Image.open(io.BytesIO(message[HEADER.size:])).convert("L")
                pixels = np.asarray(image, dtype=np.float32)
                frames[pres] = pixels[pixels.shape[0] // 2:]
            else:
                event = json.loads(message)
                if event["type"] == "chunk":
                    chunk_back[event["chunk"]] = time.perf_counter()
                elif event["type"] == "error":
                    raise SystemExit(f"server error: {event}")
                elif event["type"] == "done":
                    done = event
                    break
        await sender
        await ws.send(json.dumps({"type": "stop"}))

    # Mouth vs loudness: per frame, the audio RMS over that frame's 40 ms, and the lower face's
    # distance from its quietest frame. A mouth that follows the audio correlates positively.
    times = sorted(frames)
    hop = rate / args.fps
    loudness = np.array([float(np.sqrt(np.mean(mono[int(t / 1000 * rate): int(t / 1000 * rate + hop)] ** 2) or 0.0)) for t in times])
    quietest = frames[times[int(np.argmin(loudness))]]
    movement = np.array([float(np.abs(frames[t] - quietest).mean()) for t in times])
    # A correlation needs both series to vary; a still face (e.g. silent input) has none to measure.
    varies = len(times) > 2 and loudness.std() > 0 and movement.std() > 0
    correlation = float(np.corrcoef(loudness, movement)[0, 1]) if varies else None
    latency = [(chunk_back[k] - sent_at[k]) * 1000 for k in range(len(sent_at)) if k in chunk_back]
    return {
        "audio": args.audio, "seconds": round(len(mono) / rate, 2), "sampleRate": int(rate), "chunksSent": len(pieces),
        "chunksAnimated": len(chunk_back), "frames": len(frames), "drive": audio_ready.get("drive"), "note": audio_ready.get("note"),
        "chunkToFramesMs": {"p50": round(percentile(latency, 50), 1), "p95": round(percentile(latency, 95), 1), "max": round(max(latency), 1)},
        "mouthFollowsLoudness": {"pearson": round(correlation, 3) if correlation is not None else None,
                                 "lowerFaceMovementMax": round(float(movement.max()), 3) if len(times) else None,
                                 "method": "per-frame audio RMS vs lower-face distance from its quietest frame"},
        "serverSummary": done,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--text", default=None, help="text to speak (or --audio)")
    parser.add_argument("--audio", default=None, help="stream this recording instead of text (R-41)")
    parser.add_argument("--seconds", type=float, default=10.0, help="with --audio: how much of it to send")
    parser.add_argument("--consent-basis", default="open-licence", help="with --audio: the basis the voice is used under")
    parser.add_argument("--language", default="en")
    parser.add_argument("--mode", default="fast", choices=["fast", "clone"])
    parser.add_argument("--emotion", default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--speaker-wav", default=None)
    parser.add_argument("--clone-engine", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--save", default=None, help="folder for sample frames and the audio")
    parser.add_argument("--json-out", action="store_true", help="also write the report to outputs/benchmarks/")
    args = parser.parse_args()
    if (args.text is None) == (args.audio is None):
        parser.error("give exactly one of --text or --audio")
    report = asyncio.run(run_audio(args) if args.audio else run(args))
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = PROJECT_ROOT / "outputs" / "benchmarks"
        out.mkdir(parents=True, exist_ok=True)
        destination = out / f"live-{time.strftime('%Y%m%d-%H%M%S')}.json"
        destination.write_text(json.dumps(report, indent=2))
        print(f"written: {destination}")
    if args.audio:
        return 0 if report["chunksAnimated"] == report["chunksSent"] else 1
    return 0 if report["undecodableOrWrongSizeFrames"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
