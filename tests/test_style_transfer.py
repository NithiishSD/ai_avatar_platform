"""Style transfer (T8.6) with Stable Diffusion replaced by a fake: provenance inheritance, refusals, identity."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import provenance
import style_transfer
from avatar_store import AvatarConsentError, AvatarError, AvatarNotFound, AvatarStore
from vision_fixtures import FakeFaceEngine, gradient_image


class FakeTransfer:
    repo_id = "fake/sd"

    def __init__(self):
        self.calls = []

    def stylize(self, image, style, seed=0, steps=30):
        self.calls.append((style, seed, steps))
        return np.ascontiguousarray(255 - image[:512, :512])

    def release(self):
        pass


def passed(_image):
    return SimpleNamespace(passed=True, errors=[])


class StylizeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = AvatarStore(root=Path(self._tmp.name) / "faces", engine=FakeFaceEngine())
        self.transfer = FakeTransfer()

    def run_it(self, source="demo", style="cartoon", new="demo-cartoon", check=passed, cosine=0.71):
        return style_transfer.stylize_registered_avatar(self.store, check, source, style, new, seed=3, steps=12,
                                                        transfer=self.transfer, identity=lambda a, b: cosine)

    def test_a_synthetic_source_gives_a_synthetic_copy_that_records_its_lineage_and_identity(self):
        self.store.register(gradient_image(), avatar_id="demo", source=provenance.SYNTHETIC)
        result = self.run_it()
        self.assertEqual(self.transfer.calls, [("cartoon", 3, 12)])
        record = self.store.get("demo-cartoon")
        self.assertTrue(record.usable)
        self.assertEqual(record.provenance["source"], "synthetic")
        extra = record.provenance["extra"]
        self.assertEqual((extra["derivedFrom"], extra["style"], extra["strength"], extra["identityCosine"]), ("demo", "cartoon", 0.55, 0.71))
        self.assertEqual((result["identity"]["percent"], result["identity"]["samePerson"]), (71.0, True))

    def test_a_real_persons_photo_passes_its_subject_and_consent_basis_to_the_copy(self):
        self.store.register(gradient_image(), avatar_id="alice", source=provenance.HUMAN, subject="Alice Example",
                            consent_basis="subject-provided", licence="own photo")
        self.run_it(source="alice", new="alice-sketch", style="sketch", cosine=0.2)
        copy = self.store.get("alice-sketch").provenance
        self.assertEqual((copy["source"], copy["consentBasis"], copy["speaker"]), ("human", "subject-provided", "Alice Example"))
        self.assertTrue(self.store.get("alice-sketch").usable)

    def test_low_identity_is_reported_not_hidden(self):
        self.store.register(gradient_image(), avatar_id="demo", source=provenance.SYNTHETIC)
        result = self.run_it(cosine=0.2)
        self.assertFalse(result["identity"]["samePerson"])

    def test_a_source_without_consent_is_refused_before_any_generation(self):
        self.store.register(gradient_image(), avatar_id="demo", source=provenance.SYNTHETIC)
        provenance.sidecar_path(self.store.get("demo").path).unlink()
        with self.assertRaises(AvatarConsentError):
            self.run_it()
        self.assertEqual(self.transfer.calls, [])

    def test_a_taken_id_or_unknown_style_is_refused_before_any_generation(self):
        self.store.register(gradient_image(), avatar_id="demo", source=provenance.SYNTHETIC)
        with self.assertRaisesRegex(AvatarError, "already exists"):
            self.run_it(new="demo")
        with self.assertRaisesRegex(ValueError, "unknown style"):
            self.run_it(style="vaporwave")
        self.assertEqual(self.transfer.calls, [])

    def test_a_picture_that_fails_the_face_gate_is_not_registered(self):
        self.store.register(gradient_image(), avatar_id="demo", source=provenance.SYNTHETIC)
        failed = lambda image: SimpleNamespace(passed=False, errors=[SimpleNamespace(code="no_face")])  # noqa: E731
        with self.assertRaisesRegex(style_transfer.StyleTransferFailed, "no_face"):
            self.run_it(check=failed)
        with self.assertRaises(AvatarNotFound):
            self.store.get("demo-cartoon")

    def test_every_style_has_a_bounded_strength(self):
        for name, style in style_transfer.STYLES.items():
            self.assertTrue(0.2 <= style.strength <= 0.6, name)


if __name__ == "__main__":
    unittest.main()
