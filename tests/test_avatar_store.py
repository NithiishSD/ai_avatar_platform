"""Avatar store: the one door a face comes through (consent + quality gate)."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import provenance
from avatar_store import (
    MAX_IMAGE_SIDE,
    AvatarConsentError,
    AvatarError,
    AvatarNotFound,
    AvatarRejected,
    AvatarStore,
    decode_image,
    validate_avatar_id,
)
from vision_fixtures import FakeFaceEngine, gradient_image, synthetic_analysis


def png_bytes(size=(64, 48), exif=None) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (120, 90, 60))
    if exif:
        data = Image.Exif()
        for tag, value in exif.items():
            data[tag] = value
        image.save(buffer, format="JPEG", exif=data)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "faces"
        self.addCleanup(self._tmp.cleanup)
        self.engine = FakeFaceEngine()
        self.store = AvatarStore(root=self.root, engine=self.engine)

    def human(self, avatar_id="alice", **overrides):
        kwargs = dict(
            avatar_id=avatar_id,
            source=provenance.HUMAN,
            subject="Alice Example",
            consent_basis="subject-provided",
            licence="own photo",
        )
        kwargs.update(overrides)
        return self.store.register(gradient_image(128), **kwargs)


class AvatarIdTests(unittest.TestCase):
    def test_safe_ids_pass(self):
        for avatar_id in ("demo", "AVATAR_FEMALE_04", "a-b_c9"):
            self.assertEqual(validate_avatar_id(avatar_id), avatar_id)

    def test_path_tricks_are_refused_not_sanitised(self):
        for avatar_id in ("../x", "a/b", "", ".hidden", "a b", "x" * 65, None):
            with self.assertRaises(AvatarError):
                validate_avatar_id(avatar_id)


class DecodeTests(unittest.TestCase):
    def test_decodes_bytes_paths_and_arrays_to_rgb(self):
        array, notices = decode_image(png_bytes())
        self.assertEqual(array.shape, (48, 64, 3))
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(notices, {})
        self.assertEqual(decode_image(np.zeros((10, 12), dtype=np.uint8))[0].shape, (10, 12, 3))
        self.assertEqual(decode_image(np.zeros((10, 12, 4), dtype=np.uint8))[0].shape, (10, 12, 3))

    def test_not_an_image_says_so(self):
        with self.assertRaises(AvatarError) as ctx:
            decode_image(b"this is not an image")
        self.assertIn("not an image", str(ctx.exception))
        with self.assertRaises(AvatarError):
            decode_image("/nonexistent/photo.jpg")

    def test_large_images_are_scaled_down(self):
        array, _ = decode_image(np.zeros((100, MAX_IMAGE_SIDE * 2, 3), dtype=np.uint8))
        self.assertEqual(array.shape[1], MAX_IMAGE_SIDE)
        self.assertEqual(array.shape[0], 50)

    def test_embedded_rights_notice_is_surfaced(self):
        # The case that forced the first test photo to be deleted.
        _, notices = decode_image(png_bytes(exif={0x8298: "Official photo. May not be manipulated."}))
        self.assertIn("manipulated", notices["copyright"])


class RegistrationTests(StoreCase):
    def test_human_face_needs_a_consent_basis(self):
        with self.assertRaises(AvatarError) as ctx:
            self.human(consent_basis="")
        self.assertIn("consent basis", str(ctx.exception))
        with self.assertRaises(AvatarError):
            self.human(consent_basis="found-it-online")
        self.assertEqual(self.store.list(), [])

    def test_human_face_needs_a_named_subject(self):
        with self.assertRaises(AvatarError) as ctx:
            self.human(subject="  ")
        self.assertIn("subject", str(ctx.exception))

    def test_unknown_source_is_refused(self):
        with self.assertRaises(AvatarError):
            self.store.register(gradient_image(64), avatar_id="x", source="scraped")

    def test_registers_png_with_sidecar(self):
        record, report = self.human()
        self.assertTrue(record.usable)
        self.assertEqual(record.path.name, "alice.png")
        self.assertEqual((record.width, record.height), (128, 128))
        self.assertTrue(report.passed)
        sidecar = json.loads(provenance.sidecar_path(record.path).read_text())
        self.assertEqual(sidecar["source"], "human")
        self.assertEqual(sidecar["speaker"], "Alice Example")
        self.assertEqual(sidecar["consentBasis"], "subject-provided")
        payload = record.to_dict()
        self.assertEqual(payload["avatarId"], "alice")
        self.assertEqual(payload["imageUrl"], "/api/v1/avatar/faces/alice/image")

    def test_synthetic_face_needs_no_consent_and_keeps_its_lineage(self):
        record, _ = self.store.register(
            gradient_image(128), avatar_id="demo", source=provenance.SYNTHETIC, extra={"seed": 7}
        )
        self.assertTrue(record.usable)
        self.assertIn("no real person", record.usability_reason)
        self.assertEqual(record.provenance["extra"]["seed"], 7)
        self.assertFalse(record.provenance["admissible"])

    def test_embedded_notice_is_recorded_and_logged(self):
        with self.assertLogs("avatar_store", level="WARNING"):
            record, _ = self.store.register(
                png_bytes(size=(128, 128), exif={0x8298: "Do not manipulate"}),
                avatar_id="notice",
                source=provenance.SYNTHETIC,
            )
        self.assertEqual(record.provenance["extra"]["embeddedNotices"]["copyright"], "Do not manipulate")
        # Stored as PNG: the EXIF block itself is gone.
        with Image.open(record.path) as stored:
            self.assertEqual(stored.format, "PNG")
            self.assertEqual(len(stored.getexif()), 0)

    def test_duplicate_id_needs_overwrite(self):
        self.human()
        with self.assertRaises(AvatarError):
            self.human()
        record, _ = self.human(overwrite=True, subject="Alice Again")
        self.assertEqual(record.provenance["speaker"], "Alice Again")

    def test_quality_gate_rejects_and_writes_nothing(self):
        cases = {
            "no_face": [],
            "multiple_faces": [synthetic_analysis(), synthetic_analysis()],
            "yaw_too_large": [synthetic_analysis(yaw=41.0)],
        }
        for code, faces in cases.items():
            self.engine.faces = faces
            with self.assertRaises(AvatarRejected) as ctx:
                self.human(avatar_id=f"bad-{code.replace('_', '-')}")
            self.assertEqual([issue.code for issue in ctx.exception.report.errors], [code])
            self.assertTrue(str(ctx.exception))
        self.assertFalse(self.root.exists() and any(self.root.iterdir()))

    def test_quality_check_can_be_skipped(self):
        self.engine.faces = []
        record, report = self.human(check_quality=False)
        self.assertIsNone(report)
        self.assertTrue(record.usable)


class ConsentEnforcementTests(StoreCase):
    def test_unknown_avatar_error_names_the_fix(self):
        with self.assertRaises(AvatarNotFound) as ctx:
            self.store.get("ghost")
        self.assertIn("make_avatar.py", str(ctx.exception))

    def test_file_dropped_in_by_hand_is_listed_but_not_usable(self):
        self.root.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(self.root / "sneaked.jpg")
        records = self.store.list()
        self.assertEqual([r.avatar_id for r in records], ["sneaked"])
        self.assertFalse(records[0].usable)
        with self.assertRaises(AvatarConsentError) as ctx:
            self.store.require_usable("sneaked")
        self.assertIn("no provenance record", str(ctx.exception))
        with self.assertRaises(AvatarConsentError):
            self.store.load_image("sneaked")

    def test_sidecar_without_valid_consent_blocks_use(self):
        record, _ = self.human()
        provenance.write(record.path, source=provenance.HUMAN, speaker="Alice", consent_basis="")
        with self.assertRaises(AvatarConsentError):
            self.store.require_usable("alice")

    def test_load_image_returns_rgb_array(self):
        self.human()
        image = self.store.load_image("alice")
        self.assertEqual(image.shape, (128, 128, 3))
        np.testing.assert_array_equal(image, gradient_image(128))

    def test_delete_removes_image_and_sidecar(self):
        record, _ = self.human()
        self.store.delete("alice")
        self.assertFalse(record.path.exists())
        self.assertFalse(provenance.sidecar_path(record.path).exists())
        with self.assertRaises(AvatarNotFound):
            self.store.get("alice")

    def test_list_ignores_non_images_and_unsafe_names(self):
        self.human()
        (self.root / "notes.txt").write_text("x")
        Image.new("RGB", (8, 8)).save(self.root / "bad name.png")
        self.assertEqual([r.avatar_id for r in self.store.list()], ["alice"])

    def test_empty_store_lists_nothing(self):
        self.assertEqual(self.store.list(), [])


if __name__ == "__main__":
    unittest.main()
