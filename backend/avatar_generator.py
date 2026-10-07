"""
Synthetic avatar faces from Stable Diffusion 1.5 (task G1-02).

The consent rule leaves three ways to get a demo face: a team member's own
photo, a licensed one, or a face that belongs to nobody. This module is the
third. A generated portrait depicts no real person, so it can be animated,
committed to a benchmark and shown in a demo without anyone's permission --
and its provenance sidecar records the model, prompt and seed so it can be
regenerated exactly.

Generated faces are run through the same quality gate as uploads: diffusion
models produce turned heads, two faces and mangled mouths often enough that an
unchecked image would fail later, inside the renderer. ``generate_avatar``
walks seeds until one passes and reports every seed it rejected.

SD 1.5 in half precision needs about 3 GB of VRAM, so it is loaded only for
the call and released straight after (golden rule 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

import gpu_utils

# Imported for annotations only: face_engine pulls in MediaPipe, which this
# module does not otherwise need at import time.
if TYPE_CHECKING:
    from face_engine import FaceQualityReport
from model_registry import AVATAR_DIFFUSION_REPO, VISION_FETCH_COMMAND

logger = logging.getLogger(__name__)

REQUIRED_VRAM_MB = 3200
IMAGE_SIZE = 512

DEFAULT_PROMPT = (
    "passport style portrait photograph of an adult facing the camera, head "
    "and shoulders, wearing a plain dark sweater, open eyes looking straight "
    "into the camera, relaxed closed mouth, plain grey background, soft even "
    "lighting, sharp focus, 50mm"
)
DEFAULT_NEGATIVE = (
    "closed eyes, profile, side view, turned head, looking down, open mouth, "
    "teeth, smile, sunglasses, bare shoulders, nude, hands, two people, multiple faces, cropped head, blurry, deformed, "
    "cartoon, painting, text, watermark"
)


# Warnings a user's own photo may carry, but a generated face must not: there
# is no reason to keep a closed-eyed portrait when another seed is 10 s away.
STRICT_WARNING_CODES = frozenset({"eyes_closed", "mouth_open", "pitch_large", "roll_large"})


class AvatarGeneratorUnavailable(RuntimeError):
    """Stable Diffusion could not be loaded; the message names the fix."""


class AvatarGenerationFailed(RuntimeError):
    """No seed produced a face that passed the quality gate."""


@dataclass
class GeneratedAvatar:
    """One generated portrait and the recipe that reproduces it."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    seed: int
    prompt: str
    negative_prompt: str
    steps: int
    guidance_scale: float
    model: str = AVATAR_DIFFUSION_REPO
    rejected_seeds: Dict[int, str] = field(default_factory=dict)

    def lineage(self) -> Dict[str, object]:
        """The ``extra`` block written into the provenance sidecar."""
        return {
            "generator": self.model,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidanceScale": self.guidance_scale,
            "size": IMAGE_SIZE,
            "rejectedSeeds": {str(k): v for k, v in self.rejected_seeds.items()},
        }


class AvatarGenerator:
    """Lazy-loading SD 1.5 text-to-image with cached failure."""

    def __init__(self, repo_id: str = AVATAR_DIFFUSION_REPO, device: Optional[str] = None) -> None:
        self.repo_id = repo_id
        self.device = device
        self._pipe = None
        self._load_error: Optional[str] = None
        gpu_utils.register_releaser("avatar-diffusion", self.release)

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        if self._load_error is not None:
            raise AvatarGeneratorUnavailable(self._load_error)

        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "Stable Diffusion 1.5", keep="avatar-diffusion")
        try:
            from diffusers import StableDiffusionPipeline

            half = device == "cuda"
            # Any: diffusers' stubs type from_pretrained as a union including a
            # dummy placeholder class, which hides every real pipeline method.
            pipe: Any = StableDiffusionPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16 if half else torch.float32,
                variant="fp16",
                use_safetensors=True,
                # Never reach for the network at generation time: a missing
                # file must fail here, with the fetch command, not download.
                local_files_only=True,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_attention_slicing()
            self._pipe = pipe.to(device)
            self.device = device
            logger.info("Loaded %s on %s", self.repo_id, device)
            return self._pipe
        except gpu_utils.InsufficientVRAM:
            raise
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"Could not load {self.repo_id}: {exc}. Fetch the weights with: "
                f"{VISION_FETCH_COMMAND} --only avatar-diffusion"
            )
            raise AvatarGeneratorUnavailable(self._load_error) from exc

    def generate(
        self,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE,
        seed: int = 0,
        steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> np.ndarray:
        """One 512x512 RGB image for ``seed``."""
        import torch

        pipe = self._load()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generator,
        )
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged and flagged[0]:
            raise AvatarGenerationFailed(
                f"seed {seed}: the safety checker blanked the image"
            )
        return np.ascontiguousarray(np.asarray(result.images[0].convert("RGB"), dtype=np.uint8))

    def release(self) -> None:
        """Drop the pipeline so another heavy model can use the GPU."""
        if self._pipe is not None:
            self._pipe = None
            gpu_utils.empty_cache()


def generate_avatar(
    check: Callable[[np.ndarray], "FaceQualityReport"],
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    attempts: int = 6,
    steps: int = 30,
    guidance_scale: float = 7.0,
    generator: Optional[AvatarGenerator] = None,
) -> GeneratedAvatar:
    """
    Generate a portrait that passes the quality gate.

    ``check`` is ``FaceMeshEngine.check_quality`` (or a stand-in): it returns a
    report with ``passed`` and ``errors``. Seeds ``seed, seed+1, ...`` are
    tried in order. Errors reject an image, and so do the warnings in
    ``STRICT_WARNING_CODES``. The pipeline is released before returning, pass
    or fail.
    """
    own = generator is None
    generator = generator or AvatarGenerator()
    rejected: Dict[int, str] = {}
    tried: List[int] = []
    try:
        for offset in range(max(1, int(attempts))):
            current = int(seed) + offset
            tried.append(current)
            try:
                image = generator.generate(prompt, negative_prompt, current, steps, guidance_scale)
            except AvatarGenerationFailed as err:
                rejected[current] = str(err)
                continue
            report = check(image)
            strict = [i.code for i in report.warnings if i.code in STRICT_WARNING_CODES]
            if report.passed and not strict:
                return GeneratedAvatar(
                    image=image,
                    seed=current,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    model=generator.repo_id,
                    rejected_seeds=rejected,
                )
            reason = "; ".join([issue.code for issue in report.errors] + strict) or "rejected"
            rejected[current] = reason
            logger.warning("Generated face for seed %d rejected: %s", current, reason)
    finally:
        if own:
            generator.release()
    raise AvatarGenerationFailed(
        f"none of the seeds {tried} produced a usable face "
        f"({'; '.join(f'{k}: {v}' for k, v in rejected.items())}). "
        "Try another --seed or more --attempts."
    )
