"""Voice-to-avatar (T8.2): an uploaded recording is checked, transcribed if needed, aligned and recorded as supplied."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

import manifest
import voice_to_avatar
from audit_log import AuditLog
from voice_to_avatar import VoiceToAvatarError

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


class FakeAligner:
    last_method = "mms_fa"

    def __init__(self, stamps=None):
        self.calls = []
        self.stamps = stamps if stamps is not None else [{"phoneme": "HH", "viseme": "viseme_kk", "startMs": 100, "endMs": 300}]

    def align(self, **kw):
        self.calls.append(kw)
        return [SimpleNamespace(model_dump=lambda by_alias, s=s: dict(s)) for s in self.stamps]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class PrepareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.log = AuditLog(":memory:")
        for patcher in (mock.patch.object(voice_to_avatar, "SUPPLIED_DIR", self.dir / "supplied"),
                        mock.patch("audit_log.shared_audit", return_value=self.log)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.asr = mock.Mock()
        self.asr.transcribe.return_value = ("hello there", "en")

    def upload(self, seconds=1.0, level=0.3, rate=22050):
        path = self.dir / f"upload-{seconds}-{level}.wav"  # one file per shape: cases are built before they run
        t = np.arange(int(seconds * rate)) / rate
        sf.write(path, (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32), rate)
        return path

    def prepare(self, **kw):
        kw.setdefault("transcriber", self.asr)
        kw.setdefault("aligner", FakeAligner())
        return voice_to_avatar.prepare(kw.pop("upload", None) or self.upload(), kw.pop("basis", "speaker-recorded"), **kw)

    def test_a_given_transcript_is_used_and_no_speech_recognition_runs(self):
        speech = self.prepare(transcript="hello there", language="en")
        self.asr.transcribe.assert_not_called()
        self.assertEqual((speech.transcript, speech.transcript_source), ("hello there", "client"))
        self.assertEqual(sf.info(str(speech.audio_path)).samplerate, 24000)   # decoded to the renderer's rate
        self.assertAlmostEqual(speech.duration_seconds, 1.0, places=2)

    def test_without_a_transcript_the_words_and_language_come_from_asr(self):
        self.asr.transcribe.return_value = ("hola amigo", "es")
        aligner = FakeAligner()
        speech = self.prepare(aligner=aligner)
        self.assertEqual((speech.transcript, speech.transcript_source, speech.language), ("hola amigo", "asr", "es"))
        self.assertEqual(aligner.calls[0]["language"], "es")
        audio16k = self.asr.transcribe.call_args[0][0]
        self.assertAlmostEqual(len(audio16k) / 16000, 1.0, places=2)    # ASR gets 16 kHz

    def test_the_record_says_supplied_with_its_basis_and_the_audit_trail_has_it(self):
        speech = self.prepare()
        record = json.loads(manifest.speech_record_path(speech.audio_path).read_text())
        self.assertEqual(record["origin"], {"type": "supplied", "consentBasis": "speaker-recorded", "transcriptSource": "asr",
                                            "asrModel": voice_to_avatar.ASR_MODEL})
        self.assertEqual((record["model"], record["audioWatermark"]["applied"]), ("supplied", False))
        self.assertEqual(manifest.read_speech_record(speech.audio_path)["origin"]["type"], "supplied")
        entry = self.log.query(event="audio_supplied")[0]
        self.assertEqual((entry["subject"], entry["basis"]), (speech.sha256, "speaker-recorded"))

    def test_refusals_say_why(self):
        cases = [
            (dict(basis="i-found-it-online"), "consentBasis"),
            (dict(upload=self.upload(level=0.0)), "silent"),
            (dict(upload=self.upload(seconds=0.2)), "at least"),
            (dict(aligner=FakeAligner(stamps=[])), "could not be aligned"),
        ]
        for kwargs, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(VoiceToAvatarError, fragment):
                self.prepare(**kwargs)
        with mock.patch.object(voice_to_avatar, "MAX_SECONDS", 0.8), self.assertRaisesRegex(VoiceToAvatarError, "limit"):
            self.prepare()
        not_audio = self.dir / "x.wav"
        not_audio.write_text("not audio")
        with self.assertRaisesRegex(VoiceToAvatarError, "could not read"):
            self.prepare(upload=not_audio)

    def test_asr_that_hears_nothing_is_refused_and_leaves_no_file(self):
        self.asr.transcribe.return_value = ("", "en")
        with self.assertRaisesRegex(VoiceToAvatarError, "send the transcript"):
            self.prepare()
        self.assertEqual(list((self.dir / "supplied").glob("*.wav")), [])

    def test_timestamps_past_the_end_of_the_audio_are_dropped(self):
        stamps = [{"phoneme": "A", "viseme": "viseme_aa", "startMs": 0, "endMs": 500},
                  {"phoneme": "B", "viseme": "viseme_PP", "startMs": 900, "endMs": 1500}]
        speech = self.prepare(aligner=FakeAligner(stamps=stamps))
        self.assertEqual(len(speech.phoneme_timestamps), 1)


if __name__ == "__main__":
    unittest.main()
