"""Signed manifests and speech records (R-33, R-36): what is signed, what breaks the signature, what is withheld."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import manifest
import provenance
import watermark_engine


class ManifestCase(unittest.TestCase):
    def setUp(self):
        watermark_engine.reset_key_cache()
        self.addCleanup(watermark_engine.reset_key_cache)
        patcher = mock.patch.dict(os.environ, {"WATERMARK_KEY": "manifest-test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "clip.mp4"
        self.video.write_bytes(b"not really an mp4, but bytes are bytes" * 100)
        self.audio = self.dir / "speech.wav"
        sf.write(self.audio, np.zeros(2400, dtype=np.float32), 24000)

    def build(self, **overrides):
        args = dict(
            manifest_id=bytes(range(16)), video_path=self.video, audio_path=self.audio,
            avatar={"avatarId": "demo", "provenance": {"source": "synthetic", "consentBasis": "", "licence": "OpenRAIL-M",
                                                       "extra": {"generator": "sd15", "seed": 7}}},
            render={"width": 512, "height": 512, "fps": 25, "frameCount": 10, "durationSeconds": 0.4, "engine": "blendshape",
                    "quality": "PREVIEW", "background": None, "label": True},
            video_watermark={"applied": True, "detected": True}, models={"speech": "kokoro"},
        )
        args.update(overrides)
        with mock.patch.object(manifest, "_avatar_hash", return_value="a" * 64):
            return manifest.build_manifest(**args)


class SignatureTests(ManifestCase):
    def test_a_fresh_manifest_verifies_and_is_ours(self):
        report = manifest.verify(self.build(), self.video)
        self.assertTrue(report.valid_signature and report.issued_by_us and report.matches_file and report.trustworthy)
        self.assertEqual(report.problems, [])

    def test_editing_any_field_anywhere_breaks_the_signature(self):
        original = self.build()
        edits = [
            lambda m: m["inputs"]["avatar"].update(consentBasis="written-consent"),
            lambda m: m["content"].update(videoSha256="0" * 64),
            lambda m: m.update(aiGenerated=False),
            lambda m: m["models"].update(speech="someone-else"),
            lambda m: m["inputs"]["audio"].update(sha256="f" * 64),
            lambda m: m.update(manifestId="00" * 16),
            lambda m: m["watermarks"]["video"].update(detected=False),
            lambda m: m.update(extra="smuggled"),
        ]
        for edit in edits:
            tampered = copy.deepcopy(original)
            edit(tampered)
            report = manifest.verify(tampered)
            self.assertFalse(report.valid_signature)
            self.assertFalse(report.trustworthy)

    def test_a_stripped_or_garbage_signature_fails_without_raising(self):
        stripped = {k: v for k, v in self.build().items() if k != "signature"}
        self.assertFalse(manifest.verify(stripped).valid_signature)
        garbage = {**self.build(), "signature": {"algorithm": "ed25519", "value": "!!!not base64!!!"}}
        self.assertFalse(manifest.verify(garbage).valid_signature)
        wrong_alg = {**self.build(), "signature": {"algorithm": "rsa", "value": "AAAA"}}
        self.assertIn("unsupported", " ".join(manifest.verify(wrong_alg).problems))

    def test_not_a_manifest_at_all_is_reported_not_raised(self):
        for thing in ({}, {"schema": "something-else"}, [], "text", None):
            self.assertFalse(manifest.verify(thing).valid_signature)  # type: ignore[arg-type]

    def test_a_manifest_signed_by_someone_elses_key_is_valid_but_not_ours(self):
        with mock.patch.dict(os.environ, {"WATERMARK_KEY": "another-platform"}):
            watermark_engine.reset_key_cache()
            foreign = self.build()
        watermark_engine.reset_key_cache()  # back to our key
        report = manifest.verify(foreign, self.video)
        self.assertTrue(report.valid_signature)
        self.assertFalse(report.issued_by_us)
        self.assertFalse(report.trustworthy)
        self.assertIn("not issued here", " ".join(report.problems))

    def test_a_different_file_is_caught_by_its_hash_and_no_file_means_unknown(self):
        good = self.build()
        other = self.dir / "other.mp4"
        other.write_bytes(b"a different video")
        report = manifest.verify(good, other)
        self.assertTrue(report.valid_signature)
        self.assertFalse(report.matches_file)
        self.assertFalse(report.trustworthy)
        self.assertIsNone(manifest.verify(good).matches_file)

    def test_the_public_key_is_stable_per_secret_and_differs_between_secrets(self):
        first = manifest.public_key_hex()
        self.assertEqual(first, manifest.public_key_hex())
        self.assertEqual(len(first), 64)
        with mock.patch.dict(os.environ, {"WATERMARK_KEY": "different"}):
            watermark_engine.reset_key_cache()
            self.assertNotEqual(first, manifest.public_key_hex())

    def test_canonical_form_ignores_key_order_but_not_values(self):
        self.assertEqual(manifest.canonical({"a": 1, "b": [1, 2]}), manifest.canonical({"b": [1, 2], "a": 1}))
        self.assertNotEqual(manifest.canonical({"a": 1}), manifest.canonical({"a": 2}))

    def test_manifest_ids_are_sixteen_random_bytes(self):
        ids = {manifest.new_manifest_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(len(i) == 16 for i in ids))


class ContentTests(ManifestCase):
    def test_the_manifest_states_what_went_in_without_naming_anyone(self):
        built = self.build(avatar={"avatarId": "me", "provenance": {
            "source": "human", "consentBasis": "written-consent", "licence": "own photo", "subject": "Jane Q. Person", "speaker": "Jane Q. Person"}})
        text = json.dumps(built)
        self.assertNotIn("Jane", text)
        self.assertEqual(built["inputs"]["avatar"]["consentBasis"], "written-consent")
        self.assertEqual(built["inputs"]["avatar"]["source"], "human")
        self.assertTrue(built["aiGenerated"])
        self.assertEqual(built["content"]["videoSha256"], manifest.sha256_file(self.video))

    def test_issuer_carries_the_public_key_and_algorithm(self):
        built = self.build()
        self.assertEqual(built["issuer"]["publicKey"], manifest.public_key_hex())
        self.assertEqual(built["issuer"]["algorithm"], "ed25519")


class SpeechRecordTests(ManifestCase):
    def write(self, **kw):
        args = dict(model="kokoro", mode="fast", language="en", speaker_wav=None, clone_engine=None, emotion=None,
                    alignment_method="mms_fa", duration_seconds=0.1, watermark={"applied": True})
        args.update(kw)
        return manifest.write_speech_record(self.audio, **args)

    def test_a_record_for_these_exact_bytes_is_matched(self):
        self.write()
        record = manifest.read_speech_record(self.audio)
        self.assertEqual((record["status"], record["model"], record["audioWatermark"]), ("matched", "kokoro", {"applied": True}))

    def test_no_record_edited_audio_and_broken_json_are_each_reported_honestly(self):
        self.assertEqual(manifest.read_speech_record(self.audio)["status"], "unrecorded")
        self.write()
        sf.write(self.audio, np.ones(2400, dtype=np.float32) * 0.1, 24000)  # the file was overwritten
        self.assertEqual(manifest.read_speech_record(self.audio)["status"], "mismatched")
        manifest.speech_record_path(self.audio).write_text("{not json")
        self.assertEqual(manifest.read_speech_record(self.audio)["status"], "unreadable")

    def test_a_voice_reference_is_described_by_basis_and_hash_never_by_speaker(self):
        reference = self.dir / "voice.wav"
        sf.write(reference, np.zeros(2400, dtype=np.float32), 24000)
        provenance.write(reference, source="human", speaker="Jane Q. Person", licence="own voice", consent_basis="written-consent")
        self.write(speaker_wav=str(reference), clone_engine="xtts-v2", model="xtts-v2", mode="clone")
        record = manifest.read_speech_record(self.audio)
        self.assertNotIn("Jane", json.dumps(record))
        self.assertEqual(record["voiceReference"]["consentBasis"], "written-consent")
        self.assertEqual(record["voiceReference"]["sha256"], manifest.sha256_file(reference))
        self.assertEqual(record["cloneEngine"], "xtts-v2")

    def test_the_speech_record_ends_up_inside_the_signed_manifest(self):
        self.write()
        built = self.build()
        self.assertEqual(built["inputs"]["audio"]["speechRecord"]["status"], "matched")
        self.assertTrue(manifest.verify(built).valid_signature)


if __name__ == "__main__":
    unittest.main()
