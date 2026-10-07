"""Synthetic avatar generation (G1-02). Stable Diffusion is never loaded."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import avatar_generator
from avatar_generator import (
    AvatarGenerationFailed,
    AvatarGenerator,
    AvatarGeneratorUnavailable,
    generate_avatar,
)
from face_engine import FaceQualityReport, QualityIssue


class FakeGenerator:
    repo_id = "fake/sd"

    def __init__(self, blanked=()):
        self.seeds = []
        self.blanked = set(blanked)
        self.released = False

    def generate(self, prompt, negative_prompt, seed, steps, guidance_scale):
        self.seeds.append(seed)
        if seed in self.blanked:
            raise AvatarGenerationFailed(f"seed {seed}: the safety checker blanked the image")
        return np.full((512, 512, 3), seed % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def checker(bad=None):
    bad = bad or {}

    def check(image):
        issue = bad.get(int(image[0, 0, 0]))
        return FaceQualityReport(face_count=1, issues=[issue] if issue else [])

    return check


class PromptTests(unittest.TestCase):
    def test_attributes_shape_the_prompt(self):
        prompt = avatar_generator.build_prompt("older", "woman", "grey", glasses=True)
        for phrase in ("elderly", "woman", "short grey hair", "wearing glasses"):
            self.assertIn(phrase, prompt)
        self.assertNotIn("glasses", avatar_generator.build_prompt())

    def test_only_listed_choices_are_accepted(self):
        # Free text would let a caller name a real person; there is no way in.
        for kwargs in ({"age": "Taylor Swift"}, {"presentation": "x"}, {"hair": "x"}):
            with self.assertRaises(KeyError):
                avatar_generator.build_prompt(**kwargs)


class GenerateRegisteredTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        from avatar_store import AvatarStore
        from vision_fixtures import FakeFaceEngine

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = AvatarStore(root=Path(self._tmp.name), engine=FakeFaceEngine())

    def test_registers_as_synthetic_with_the_recipe(self):
        generator = FakeGenerator()
        avatar_generator.generate_registered_avatar(self.store, checker(), "gen1", seed=9, generator=generator)
        record = self.store.get("gen1")
        self.assertTrue(record.usable)
        self.assertEqual(record.provenance["source"], "synthetic")
        self.assertEqual(record.provenance["extra"]["seed"], 9)

    def test_existing_id_is_refused_before_any_generation(self):
        from avatar_store import AvatarError

        avatar_generator.generate_registered_avatar(self.store, checker(), "gen1", generator=FakeGenerator())
        second = FakeGenerator()
        with self.assertRaises(AvatarError):
            avatar_generator.generate_registered_avatar(self.store, checker(), "gen1", generator=second)
        self.assertEqual(second.seeds, [])  # the slow part never ran
        avatar_generator.generate_registered_avatar(self.store, checker(), "gen1", overwrite=True, generator=FakeGenerator())


class GenerateAvatarTests(unittest.TestCase):
    def test_first_good_seed_wins_and_lineage_is_reproducible(self):
        generator = FakeGenerator()
        result = generate_avatar(checker(), seed=7, generator=generator)
        self.assertEqual(result.seed, 7)
        self.assertEqual(generator.seeds, [7])
        lineage = result.lineage()
        self.assertEqual(lineage["generator"], "fake/sd")
        self.assertEqual(lineage["seed"], 7)
        self.assertEqual(lineage["prompt"], avatar_generator.DEFAULT_PROMPT)
        self.assertEqual(lineage["rejectedSeeds"], {})

    def test_rejected_seeds_are_skipped_and_recorded(self):
        bad = {
            3: QualityIssue("yaw_too_large", "turned"),
            4: QualityIssue("eyes_closed", "closed", severity="warning"),
        }
        generator = FakeGenerator(blanked={5})
        result = generate_avatar(checker(bad), seed=3, attempts=6, generator=generator)
        self.assertEqual(result.seed, 6)
        self.assertEqual(generator.seeds, [3, 4, 5, 6])
        self.assertEqual(result.rejected_seeds[3], "yaw_too_large")
        self.assertEqual(result.rejected_seeds[4], "eyes_closed")  # a warning, but strict for generated faces
        self.assertIn("safety checker", result.rejected_seeds[5])

    def test_a_tolerable_warning_does_not_reject(self):
        bad = {1: QualityIssue("something_minor", "meh", severity="warning")}
        self.assertEqual(generate_avatar(checker(bad), seed=1, generator=FakeGenerator()).seed, 1)

    def test_running_out_of_attempts_is_an_error_with_the_reasons(self):
        bad = {s: QualityIssue("no_face", "none") for s in range(10)}
        with self.assertRaises(AvatarGenerationFailed) as ctx:
            generate_avatar(checker(bad), seed=0, attempts=3, generator=FakeGenerator())
        self.assertIn("[0, 1, 2]", str(ctx.exception))
        self.assertIn("no_face", str(ctx.exception))

    def test_own_pipeline_is_released_even_on_failure(self):
        created = FakeGenerator(blanked={0})
        with mock.patch("avatar_generator.AvatarGenerator", return_value=created):
            with self.assertRaises(AvatarGenerationFailed):
                generate_avatar(checker(), seed=0, attempts=1)
        self.assertTrue(created.released)


class GeneratorLoadTests(unittest.TestCase):
    def test_missing_weights_name_the_fetch_command_and_failure_is_cached(self):
        generator = AvatarGenerator(repo_id="nobody/not-cached", device="cpu")
        fake = mock.Mock()
        fake.StableDiffusionPipeline.from_pretrained.side_effect = OSError("not in cache")
        with mock.patch.dict("sys.modules", {"diffusers": fake}):
            for _ in range(2):
                with self.assertRaises(AvatarGeneratorUnavailable) as ctx:
                    generator._load()
                self.assertIn("fetch_vision_models.py --only avatar-diffusion", str(ctx.exception))
        self.assertEqual(fake.StableDiffusionPipeline.from_pretrained.call_count, 1)
        kwargs = fake.StableDiffusionPipeline.from_pretrained.call_args.kwargs
        self.assertTrue(kwargs["local_files_only"])  # never downloads at generation time

    def test_safety_checker_blank_is_a_failure_not_a_black_avatar(self):
        from PIL import Image

        generator = AvatarGenerator(device="cpu")
        generator._pipe = mock.Mock(
            return_value=SimpleNamespace(images=[Image.new("RGB", (512, 512))], nsfw_content_detected=[True])
        )
        with self.assertRaises(AvatarGenerationFailed):
            generator.generate(seed=1)

    def test_generate_returns_rgb_array_and_release_drops_the_pipeline(self):
        from PIL import Image

        generator = AvatarGenerator(device="cpu")
        generator._pipe = mock.Mock(
            return_value=SimpleNamespace(images=[Image.new("RGB", (512, 512), (9, 8, 7))], nsfw_content_detected=[False])
        )
        image = generator.generate(seed=3, steps=5)
        self.assertEqual(image.shape, (512, 512, 3))
        self.assertEqual(tuple(image[0, 0]), (9, 8, 7))
        generator.release()
        self.assertIsNone(generator._pipe)


if __name__ == "__main__":
    unittest.main()
