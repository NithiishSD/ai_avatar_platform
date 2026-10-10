"""
Live interactive avatar (R-18, R-19): the engine's ordering and pipelining, and the
WebSocket protocol around it. A fake session stands in for the models, so these
tests load nothing; what they pin down is the contract a client relies on.
"""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module
import avatar_store
import live_engine
from live_engine import KIND_AUDIO, KIND_FRAME, ChunkAudio, LiveError, pack_media, split_sentences, unpack_media
from security import SecurityConfig, SecurityGate


# --------------------------------------------------------------------------- helpers
class FakeTrack:
    def __init__(self, frames: int, chunk: int) -> None:
        self.frame_count = frames
        self.chunk = chunk


class FakeSession:
    """Same surface as LiveSession; 0.5 s of audio per sentence, 25 fps."""

    instances: ClassVar[list] = []
    open_error: Exception | None = None
    frame_delay = 0.0

    def __init__(self, router, store, avatar_id, **options) -> None:
        self.avatar_id, self.options = avatar_id, options
        self.session_id = f"fake{len(FakeSession.instances)}"
        self.fps = options.get("fps", 25)
        self.timeline_ms = 0.0
        self.log: list = []
        self.cleaned = False
        FakeSession.instances.append(self)

    def open(self):
        if FakeSession.open_error:
            raise FakeSession.open_error
        return {"width": 384, "height": 384, "fps": self.fps}

    def synth_chunk(self, index, text) -> ChunkAudio:
        self.log.append(("synth", index, 0, time.perf_counter()))
        return ChunkAudio(
            index=index, text=text, pcm16=b"\x01\x00" * 12000, sample_rate=24000, duration_seconds=0.5,
            timestamps=[{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 400}],
            alignment_method="mms_fa", model="kokoro", emotion_vector={}, energy=np.ones(10), synth_ms=7.0,
        )

    def build_track(self, chunk):
        return FakeTrack(13, chunk.index)

    def audio_track(self, index, pcm16, sample_rate):
        self.log.append(("audio", index, len(pcm16), sample_rate))
        return FakeTrack(int(round(len(pcm16) / 2 / sample_rate * self.fps)), index)

    def render_jpeg(self, track, index) -> bytes:
        self.log.append(("frame", track.chunk, index, time.perf_counter()))
        if FakeSession.frame_delay:
            time.sleep(FakeSession.frame_delay)
        return b"\xff\xd8JPEG" + bytes([index])

    def cleanup(self):
        self.cleaned = True


def collect(session, text, **kw):
    async def run():
        return [event async for event in live_engine.stream_text(session, text, **kw)]

    return asyncio.run(run())


# --------------------------------------------------------------------------- engine
class SplitAndFramingTests(unittest.TestCase):
    def test_sentences_split_on_punctuation_including_danda_and_cjk(self):
        self.assertEqual(split_sentences("Hello there. How are you? Fine!"), ["Hello there.", "How are you?", "Fine!"])
        self.assertEqual(split_sentences("यह पहला है। यह दूसरा है।"), ["यह पहला है।", "यह दूसरा है।"])
        self.assertEqual(split_sentences("这是第一句。 这是第二句。"), ["这是第一句。", "这是第二句。"])

    def test_a_run_on_sentence_is_cut_at_a_space_and_nothing_is_lost(self):
        text = " ".join(["word"] * 200)
        pieces = split_sentences(text, max_chars=100)
        self.assertTrue(all(len(p) <= 100 for p in pieces))
        self.assertEqual(" ".join(pieces).split(), text.split())

    def test_pieces_with_nothing_to_say_are_dropped(self):
        self.assertEqual(split_sentences("... . Hello."), ["Hello."])
        self.assertEqual(split_sentences("   "), [])

    def test_media_header_round_trips(self):
        message = pack_media(KIND_FRAME, 3, 17, 1234, b"payload")
        self.assertEqual(len(message), 13 + 7)
        self.assertEqual(unpack_media(message), (KIND_FRAME, 3, 17, 1234, b"payload"))


class StreamTextTests(unittest.TestCase):
    def setUp(self):
        FakeSession.frame_delay = 0.0

    def test_events_come_in_order_with_timeline_positions(self):
        session = FakeSession(None, None, "demo")
        events = collect(session, "One. Two. Three.")
        kinds = [e.kind for e in events]
        per_chunk = ["audio"] + ["frame"] * 13 + ["chunk"]
        self.assertEqual(kinds, per_chunk * 3 + ["done"])
        audio = [unpack_media(e.payload) for e in events if e.kind == "audio"]
        self.assertEqual([a[0] for a in audio], [KIND_AUDIO] * 3)
        self.assertEqual([a[3] for a in audio], [0, 500, 1000])  # each sentence starts where the last ended
        first_frames = [unpack_media(e.payload) for e in events if e.kind == "frame"][:3]
        self.assertEqual([f[3] for f in first_frames], [0, 40, 80])  # 25 fps
        self.assertEqual(events[-1].meta["chunks"], 3)
        self.assertEqual(events[-2].meta["alignmentMethod"], "mms_fa")

    def test_the_next_sentence_is_synthesised_while_the_current_frames_render(self):
        FakeSession.frame_delay = 0.01
        session = FakeSession(None, None, "demo")
        collect(session, "One. Two.")
        synth_second = next(t for kind, chunk, _, t in session.log if kind == "synth" and chunk == 1)
        first_sentence_frames = [t for kind, chunk, _, t in session.log if kind == "frame" and chunk == 0]
        self.assertEqual(len(first_sentence_frames), 13)
        # The second sentence began synthesising before the first sentence's LAST frame was drawn:
        # the work overlapped instead of queueing behind the frames.
        self.assertLess(synth_second, max(first_sentence_frames))

    def test_text_with_no_words_is_an_error_not_a_silent_session(self):
        with self.assertRaises(LiveError):
            collect(FakeSession(None, None, "demo"), "... !!!")

    def test_audio_only_mode_sends_no_frames(self):
        events = collect(FakeSession(None, None, "demo"), "One.", with_frames=False)
        self.assertEqual([e.kind for e in events], ["audio", "chunk", "done"])

    def test_closing_early_does_not_start_more_sentences(self):
        session = FakeSession(None, None, "demo")

        async def run():
            gen = live_engine.stream_text(session, "One. Two. Three. Four.")
            await gen.__anext__()  # first audio only
            await gen.aclose()
            await asyncio.sleep(0.1)

        asyncio.run(run())
        started = {chunk for kind, chunk, _, _ in session.log if kind == "synth"}
        self.assertLessEqual(started, {0, 1})  # the one prefetched sentence, never 2 or 3

    def test_the_summary_reports_both_latencies(self):
        done = collect(FakeSession(None, None, "demo"), "One.")[-1].meta
        for key in ("textToFirstAudioMs", "textToFirstFrameMs", "firstFrameAfterFirstAudioMs", "totalMs"):
            self.assertIsNotNone(done[key], key)


class RealSessionSynthesisTests(unittest.TestCase):
    """LiveSession.synth_chunk against a fake router that writes a real wav."""

    def make(self, timestamps, tmp):
        def synthesize(text, **kwargs):
            path = Path(tmp) / "out.wav"
            sf.write(path, (0.5 * np.sin(np.linspace(0, 200, 12000))).astype(np.float32), 24000)
            return SimpleNamespace(output_path=str(path), phoneme_timestamps=timestamps, alignment_method="mms_fa",
                                   model="kokoro", emotion={"vector": {"joy": 1.0}})

        router = SimpleNamespace(synthesize=mock.Mock(side_effect=synthesize))
        return live_engine.LiveSession(router, None, "demo", emotion="joy"), router

    def test_audio_is_converted_to_pcm16_and_the_file_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, router = self.make([{"phoneme": "AA", "viseme": "viseme_aa", "startMs": 0, "endMs": 300}], tmp)
            chunk = session.synth_chunk(0, "Hello.")
            self.assertEqual(len(chunk.pcm16), 12000 * 2)
            self.assertAlmostEqual(chunk.duration_seconds, 0.5, places=3)
            self.assertFalse((Path(tmp) / "out.wav").exists())
            self.assertEqual(router.synthesize.call_args.kwargs["emotion"], "joy")
            self.assertTrue(router.synthesize.call_args.kwargs["return_alignment"])
            self.assertGreater(chunk.emotion_vector["happy"], 0)

    def test_speech_without_phoneme_timing_is_refused_with_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _ = self.make([], tmp)
            with self.assertRaisesRegex(LiveError, "no phoneme timing"):
                session.synth_chunk(0, "Hello.")


# --------------------------------------------------------------------------- websocket
def tone(seconds, level, rate=16000):
    """PCM16 of a 200 Hz tone at ``level`` (0..1 of full scale)."""
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 200 * t) * level * 32767).astype("<i2").tobytes()


class AudioDrivenTests(unittest.TestCase):
    """Streaming audio input (R-41): the client's own speech moves the mouth by its loudness."""

    def session(self):
        real = live_engine.LiveSession(mock.MagicMock(), mock.MagicMock(), "demo", fps=25)
        real.session_id = "audio-test"
        return real

    def jaw(self, track):
        return float(track.column("jawOpen").mean())

    def test_loud_speech_opens_the_mouth_and_silence_keeps_it_shut(self):
        session = self.session()
        loud = session.audio_track(0, tone(1.0, 0.5), 16000)
        silent = session.audio_track(1, tone(1.0, 0.0), 16000)
        self.assertEqual(loud.frame_count, 25)
        self.assertGreater(self.jaw(loud), 0.05)
        self.assertLess(self.jaw(silent), 1e-3)

    def test_a_quiet_chunk_after_a_loud_one_is_drawn_quieter(self):
        session = self.session()
        loud = session.audio_track(0, tone(1.0, 0.5), 16000)
        quiet = session.audio_track(1, tone(1.0, 0.08), 16000)
        self.assertLess(self.jaw(quiet), self.jaw(loud))

    def test_room_noise_alone_does_not_open_the_mouth(self):
        session = self.session()
        hiss = (np.random.default_rng(0).normal(scale=0.002, size=16000) * 32767).astype("<i2").tobytes()
        self.assertLess(self.jaw(session.audio_track(0, hiss, 16000)), 1e-3)

    def test_bad_chunks_are_refused_with_the_reason(self):
        with self.assertRaisesRegex(LiveError, "empty"):
            live_engine.check_pcm(b"", 16000)
        with self.assertRaisesRegex(LiveError, "even number"):
            live_engine.check_pcm(b"\x00" * 3, 16000)
        with self.assertRaisesRegex(LiveError, "at most 2 s"):
            live_engine.check_pcm(tone(2.5, 0.1), 16000)
        live_engine.check_pcm(tone(2.0, 0.1), 16000)

    def test_a_chunk_streams_its_frames_on_the_session_timeline_then_a_labelled_summary(self):
        fake = FakeSession(None, None, "demo")

        async def run():
            return [event async for event in live_engine.stream_audio_chunk(fake, 0, tone(0.4, 0.3), 16000)]

        events = asyncio.run(run())
        self.assertEqual([e.kind for e in events], ["frame"] * 10 + ["chunk"])
        self.assertEqual([unpack_media(e.payload)[3] for e in events[:3]], [0, 40, 80])
        self.assertEqual(events[-1].meta["drive"], "audio-energy")
        self.assertIn("not phoneme-aligned", events[-1].meta["note"])
        self.assertEqual(fake.timeline_ms, 400.0)


class AudioTimelineTests(unittest.TestCase):
    def test_half_second_chunks_keep_exactly_the_session_frame_rate_on_one_grid(self):
        fake = FakeSession(None, None, "demo")

        async def run():
            events = []
            for k in range(4):  # 2 s of audio in 0.5 s chunks: 12.5 frames each
                events += [e async for e in live_engine.stream_audio_chunk(fake, k, tone(0.5, 0.3), 16000)]
            return events

        frames = [unpack_media(e.payload)[3] for e in asyncio.run(run()) if e.kind == "frame"]
        self.assertEqual(len(frames), 50)                          # 25 fps, not 26
        self.assertEqual(frames, [i * 40 for i in range(50)])      # one grid, no repeats, no gaps


class LiveWebSocketTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances, FakeSession.open_error, FakeSession.frame_delay = [], None, 0.0
        router = mock.MagicMock()
        router.select_model.return_value = "kokoro"
        self.router = router
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for patcher in (
            mock.patch.object(app_module.live_engine, "LiveSession", FakeSession),
            mock.patch.object(app_module, "get_router", return_value=router),
            mock.patch.object(app_module.security_gate, "inspect", return_value=(True, 0, {})),
            mock.patch.object(app_module, "inputs_dir", Path(self._tmp.name)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        app_module._live_active = 0
        self.client = TestClient(app_module.app)

    def start(self, ws, **fields):
        ws.send_json({"type": "start", "avatarId": "demo", **fields})
        return ws.receive_json()

    def until_done(self, ws):
        messages = []
        while True:
            message = ws.receive()
            if message.get("text") is not None:
                parsed = __import__("json").loads(message["text"])
                messages.append(parsed)
                if parsed["type"] in ("done", "error", "interrupted"):
                    return messages
            else:
                messages.append(unpack_media(message["bytes"]))

    def test_start_say_and_the_full_message_sequence(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            ready = self.start(ws, fps=20, maxSide=256)
            self.assertEqual((ready["type"], ready["width"], ready["fps"], ready["sampleRate"]), ("ready", 384, 20, 24000))
            self.assertEqual(ready["model"], "kokoro")
            ws.send_json({"type": "say", "text": "First one. Second one."})
            messages = self.until_done(ws)
        kinds = [m["type"] if isinstance(m, dict) else ("audio-bin" if m[0] == KIND_AUDIO else "frame-bin") for m in messages]
        one_chunk = ["audio", "audio-bin"] + ["frame-bin"] * 13 + ["chunk"]
        self.assertEqual(kinds, one_chunk * 2 + ["done"])
        self.assertEqual(messages[0]["sampleRate"], 24000)
        self.assertEqual(messages[-1]["chunks"], 2)
        self.assertEqual(FakeSession.instances[0].options["fps"], 20)

    def test_a_session_can_speak_more_than_once(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            for text in ("One.", "Two."):
                ws.send_json({"type": "say", "text": text})
                self.assertEqual(self.until_done(ws)[-1]["type"], "done")
        starts = [e for e in FakeSession.instances[0].log if e[0] == "synth"]
        self.assertEqual(len(starts), 2)
        self.assertGreater(FakeSession.instances[0].timeline_ms, 0)  # the timeline kept running

    def test_bad_first_messages_are_refused_and_say_what_was_wrong(self):
        for first in ({"type": "say", "text": "hi"}, {"type": "start"}, {"type": "start", "avatarId": "../x"},
                      {"type": "start", "avatarId": "demo", "fps": 500}, {"type": "start", "avatarId": "demo", "extra": 1}):
            with self.client.websocket_connect("/api/v1/live") as ws:
                ws.send_json(first)
                reply = ws.receive_json()
            self.assertEqual((reply["type"], reply["code"]), ("error", "bad_start"), first)

    def test_a_key_is_required_when_auth_is_on_and_accepted_in_the_start_message(self):
        gate = SecurityGate(SecurityConfig(auth_enabled=True, api_keys=frozenset({"k1"}), rate_limit_enabled=False,
                                           requests_per_minute=120, burst=10))
        with mock.patch.object(app_module, "security_gate", gate):
            with self.client.websocket_connect("/api/v1/live") as ws:
                self.assertEqual(self.start(ws)["code"], "unauthorised")
            with self.client.websocket_connect("/api/v1/live") as ws:
                self.assertEqual(self.start(ws, apiKey="wrong")["code"], "unauthorised")
            with self.client.websocket_connect("/api/v1/live") as ws:
                self.assertEqual(self.start(ws, apiKey="k1")["type"], "ready")
            with self.client.websocket_connect("/api/v1/live", headers={"X-API-Key": "k1"}) as ws:
                self.assertEqual(self.start(ws)["type"], "ready")

    def test_sessions_over_the_cap_are_told_to_retry_and_the_slot_is_freed_on_close(self):
        with mock.patch.object(app_module, "LIVE_MAX_SESSIONS", 1):
            with self.client.websocket_connect("/api/v1/live") as first:
                self.assertEqual(self.start(first)["type"], "ready")
                with self.client.websocket_connect("/api/v1/live") as second:
                    reply = self.start(second)
                self.assertEqual(reply["code"], "busy")
            time.sleep(0.2)
            with self.client.websocket_connect("/api/v1/live") as third:
                self.assertEqual(self.start(third)["type"], "ready")
        time.sleep(0.2)
        self.assertEqual(app_module._live_active, 0)

    def test_a_face_without_consent_or_a_missing_avatar_is_an_error_message(self):
        for error, code in ((avatar_store.AvatarConsentError("no consent basis"), "consent"),
                            (avatar_store.AvatarNotFound("no such avatar"), "avatar_not_found")):
            FakeSession.open_error = error
            with self.client.websocket_connect("/api/v1/live") as ws:
                reply = self.start(ws)
            self.assertEqual(reply["code"], code)
            self.assertIn(str(error), reply["detail"])
        time.sleep(0.2)
        self.assertEqual(app_module._live_active, 0)  # a refused start does not leak a slot

    def test_clone_mode_needs_a_recording_inside_inputs(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.assertEqual(self.start(ws, mode="clone")["code"], "bad_request")
        replies = []
        for raw in ("/etc/passwd", "/nope/missing.wav", "../../etc/hostname"):
            with self.client.websocket_connect("/api/v1/live") as ws:
                replies.append(self.start(ws, mode="clone", speakerWav=raw))
        self.assertEqual({r["code"] for r in replies}, {"bad_request"})
        self.assertEqual(len({r["detail"] for r in replies}), 1)  # no existence oracle here either

    def test_a_clone_session_passes_the_resolved_recording_to_the_session(self):
        wav = Path(self._tmp.name) / "voice.wav"
        sf.write(wav, np.zeros(2400, dtype=np.float32), 24000)
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.assertEqual(self.start(ws, mode="clone", speakerWav=str(wav), cloneEngine="openvoice-v2")["type"], "ready")
        self.assertEqual(FakeSession.instances[0].options["speaker_wav"], str(wav.resolve()))
        self.router.preflight.assert_called_once()  # consent and weights are checked before any speech

    def test_bad_say_messages_are_reported_and_the_session_carries_on(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            for bad in ({"type": "say", "text": ""}, {"type": "say", "text": "x" * 2001}, {"type": "dance"}, {"type": "say"}):
                ws.send_json(bad)
                self.assertEqual(ws.receive_json()["code"], "bad_message", bad)
            ws.send_text("this is not json")
            self.assertEqual(ws.receive_json()["code"], "malformed")
            ws.send_json({"type": "say", "text": "Still alive."})
            self.assertEqual(self.until_done(ws)[-1]["type"], "done")

    def test_text_with_no_words_is_an_error_message_not_a_hang(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            ws.send_json({"type": "say", "text": "... !!!"})
            reply = ws.receive_json()
        self.assertEqual((reply["type"], reply["code"]), ("error", "speech_failed"))

    def test_interrupt_stops_the_speech_and_the_session_keeps_working(self):
        FakeSession.frame_delay = 0.01
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            ws.send_json({"type": "say", "text": "A long first sentence. And a second. And a third. And a fourth."})
            ws.receive_json()          # audio meta
            ws.receive_bytes()         # audio chunk
            ws.receive_bytes()         # first frame: speech is under way
            ws.send_json({"type": "interrupt"})
            tail = self.until_done(ws)
            self.assertEqual(tail[-1]["type"], "interrupted")
            frames_after = sum(1 for m in tail if isinstance(m, tuple))
            self.assertLess(frames_after, 13 * 4)  # nowhere near all four sentences' frames
            ws.send_json({"type": "say", "text": "After the interruption."})
            self.assertEqual(self.until_done(ws)[-1]["type"], "done")

    def test_stop_ends_the_session_and_cleans_up(self):
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            ws.send_json({"type": "stop"})
        time.sleep(0.2)
        self.assertTrue(FakeSession.instances[0].cleaned)
        self.assertEqual(app_module._live_active, 0)

    def test_a_client_that_vanishes_mid_speech_leaves_nothing_running(self):
        FakeSession.frame_delay = 0.02
        with self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            ws.send_json({"type": "say", "text": "One. Two. Three. Four. Five. Six."})
            ws.receive_json()
            ws.receive_bytes()
        time.sleep(0.5)
        synthesised = {chunk for kind, chunk, _, _ in FakeSession.instances[0].log if kind == "synth"}
        self.assertLess(len(synthesised), 6)  # it stopped; it did not synthesise the whole text for nobody
        self.assertTrue(FakeSession.instances[0].cleaned)
        self.assertEqual(app_module._live_active, 0)

    def test_streamed_audio_is_animated_labelled_and_logged_with_its_consent_basis(self):
        from audit_log import AuditLog

        log = AuditLog(":memory:")
        with mock.patch("audit_log.shared_audit", return_value=log), self.client.websocket_connect("/api/v1/live") as ws:
            self.start(ws)
            ws.send_bytes(tone(0.4, 0.3))                           # before audio_start: refused, session lives on
            self.assertEqual(ws.receive_json()["code"], "audio_not_started")
            ws.send_json({"type": "audio_start", "sampleRate": 16000})  # no consent basis
            self.assertEqual(ws.receive_json()["code"], "bad_message")
            ws.send_json({"type": "audio_start", "sampleRate": 16000, "consentBasis": "speaker-recorded"})
            ready = ws.receive_json()
            self.assertEqual((ready["type"], ready["drive"]), ("audio_ready", "audio-energy"))
            for _ in range(2):
                ws.send_bytes(tone(0.4, 0.3))
            ws.send_bytes(tone(3.0, 0.3))                           # too long for one chunk
            ws.send_json({"type": "audio_end"})
            messages = []
            while True:  # unlike until_done, read past the error to the final "done"
                message = ws.receive()
                if message.get("text") is None:
                    messages.append(unpack_media(message["bytes"]))
                    continue
                messages.append(__import__("json").loads(message["text"]))
                if messages[-1]["type"] == "done":
                    break
        frames = [m for m in messages if isinstance(m, tuple)]
        chunks = [m for m in messages if isinstance(m, dict) and m["type"] == "chunk"]
        errors = [m for m in messages if isinstance(m, dict) and m["type"] == "error"]
        self.assertEqual(len(frames), 20)
        self.assertEqual([c["startMs"] for c in chunks], [0.0, 400.0])     # one continuous timeline
        self.assertTrue(all(c["drive"] == "audio-energy" for c in chunks))
        self.assertEqual(messages[-1], {"type": "done", "drive": "audio-energy", "chunks": 2})
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["code"], "bad_audio")
        supplied = log.query(event="audio_supplied")
        self.assertEqual(len(supplied), 1)
        self.assertEqual((supplied[0]["basis"], supplied[0]["details"]["sampleRate"]), ("speaker-recorded", 16000))

    def test_the_connection_carries_a_request_id_in_the_ready_message(self):
        with self.client.websocket_connect("/api/v1/live", headers={"X-Request-ID": "trace-live-1"}) as ws:
            self.assertEqual(self.start(ws)["requestId"], "trace-live-1")


if __name__ == "__main__":
    unittest.main()
