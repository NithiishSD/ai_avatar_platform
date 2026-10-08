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

Where this sits: it is the very start of the vision side. ``POST
/api/v1/avatar/generate`` (``app.py``) and ``scripts/make_avatar.py`` call
``generate_registered_avatar``, which stores the face in ``AvatarStore``. From
then on the face is used like any uploaded photo: ``face_engine`` finds its
landmarks and the renderer animates it.

Concepts used throughout, explained once here:

**Stable Diffusion** is a text-to-image model. It starts from random noise and
removes a little of it per *step*, steered by the text prompt, until an image
is left. ``diffusers`` packages the parts (text encoder, U-Net denoiser, VAE
decoder, safety checker) as one *pipeline* object you call like a function.

**The seed** fixes the starting noise. Same model + prompt + seed + steps +
guidance gives the same image, which is why those five values are the
"recipe" saved in the provenance sidecar.

**Negative prompt and guidance scale.** Each step the model predicts twice:
once for the prompt and once for the negative prompt. The guidance scale says
how far to push the result away from the negative and towards the prompt.
Higher follows the text more closely but looks less natural; about 7 is the
usual SD 1.5 setting.

**A provenance sidecar** is a small JSON file stored next to a face or voice
that says where it came from (human or synthetic, licence, notes). Golden
rule 3 forbids using a face without one; ``provenance.py`` defines it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

import gpu_utils

# Imported for annotations only: face_engine pulls in MediaPipe, which this
# module does not otherwise need at import time.
# TYPE_CHECKING is False when the program runs and True only for a type
# checker, so these imports cost nothing at run time. That is also why the
# annotations below that use them are written as strings.
if TYPE_CHECKING:
    from avatar_store import AvatarStore
    from face_engine import FaceQualityReport
from model_registry import AVATAR_DIFFUSION_REPO, VISION_FETCH_COMMAND

logger = logging.getLogger(__name__)

# GPU memory that must be free before loading on CUDA: the fp16 pipeline
# needs about 3 GB (module docstring), rounded up.
REQUIRED_VRAM_MB = 3200
# Measured 8 Oct 2026: fp32 on the CPU adds 6610 MiB to the process peak.
RAM_MB = 6700
# SD 1.5 was trained on 512x512 images. Other sizes work but tend to repeat
# or stretch features, so the face is generated at its native size.
IMAGE_SIZE = 512

# The prompt used when a caller gives none (``scripts/make_avatar.py``). It asks
# for exactly what the quality gate and the lip-sync renderer want: a frontal,
# evenly lit face with open eyes and a closed mouth on a plain background.
DEFAULT_PROMPT = (
    "passport style portrait photograph of an adult facing the camera, head "
    "and shoulders, wearing a plain dark sweater, open eyes looking straight "
    "into the camera, relaxed closed mouth, plain grey background, soft even "
    "lighting, sharp focus, 50mm"
)
# Things to steer away from. Most are the quality-gate failures named in the
# module docstring (turned heads, two faces, open mouths); the rest keep the
# portrait clothed, photographic and free of text.
DEFAULT_NEGATIVE = (
    "closed eyes, profile, side view, turned head, looking down, open mouth, "
    "teeth, smile, sunglasses, bare shoulders, nude, hands, two people, multiple faces, cropped head, blurry, deformed, "
    "cartoon, painting, text, watermark"
)


# What a caller may ask for. These are fixed choices, not free text, on purpose:
# a free prompt could name a real person, and a face generated from that name
# would be a likeness of someone who never consented (golden rule 3). Every
# phrase below describes a kind of face, never an individual.
# The keys are what the API accepts (``app.py`` lists them for the UI); the
# values are the words that go into the prompt.
AGES = {"young": "young adult", "adult": "adult", "middle-aged": "middle-aged", "older": "elderly"}
PRESENTATIONS = {"person": "person", "man": "man", "woman": "woman"}
HAIR = {
    "short-dark": "short dark hair",
    "short-light": "short light brown hair",
    "long-dark": "long dark hair",
    "long-light": "long blonde hair",
    "grey": "short grey hair",
    "bald": "a shaved head",
}


def build_prompt(age: str = "adult", presentation: str = "person", hair: str = "short-dark", glasses: bool = False) -> str:
    """The diffusion prompt for a set of attribute choices (raises KeyError on an unknown one)."""
    # Dictionary lookups rather than string formatting of the raw values: an
    # unknown key raises instead of reaching the prompt, so only the vetted
    # phrases above can ever be sent to the model. The rest of the sentence
    # matches DEFAULT_PROMPT so attribute faces pass the gate as often.
    return (
        f"passport style portrait photograph of {AGES[age]} {PRESENTATIONS[presentation]} "
        f"with {HAIR[hair]}{', wearing glasses' if glasses else ''}, facing the camera, head "
        "and shoulders, wearing a plain dark sweater, open eyes looking straight "
        "into the camera, relaxed closed mouth, plain grey background, soft even "
        "lighting, sharp focus, 50mm"
    )


# Warnings a user's own photo may carry, but a generated face must not: there
# is no reason to keep a closed-eyed portrait when another seed is 10 s away.
# The codes are the ones ``face_engine.assess_face_quality`` emits. Pitch is
# the head nodding up or down; roll is the head tilting towards a shoulder.
# A frozenset is an immutable set: fast ``in`` checks, and no caller can
# accidentally add to it.
STRICT_WARNING_CODES = frozenset({"eyes_closed", "mouth_open", "pitch_large", "roll_large"})


class AvatarGeneratorUnavailable(RuntimeError):
    """Stable Diffusion could not be loaded; the message names the fix."""


class AvatarGenerationFailed(RuntimeError):
    """No seed produced a face that passed the quality gate."""


@dataclass
class GeneratedAvatar:
    """
    One generated portrait and the recipe that reproduces it.

    Everything except ``image`` and ``rejected_seeds`` is an input to the
    pipeline; together they regenerate the same pixels. ``rejected_seeds``
    maps each seed that failed to the reason, so a reader can see how many
    tries the face took.
    """

    image: np.ndarray  # (H, W, 3) uint8 RGB
    seed: int
    prompt: str
    negative_prompt: str
    steps: int
    guidance_scale: float
    model: str = AVATAR_DIFFUSION_REPO
    # field(default_factory=dict): a dataclass default must not be a shared
    # mutable object, or every instance would write into the same dict. The
    # factory builds a fresh one per instance.
    rejected_seeds: Dict[int, str] = field(default_factory=dict)

    def lineage(self) -> Dict[str, object]:
        """The ``extra`` block written into the provenance sidecar."""
        # camelCase keys because the sidecar is JSON read by the frontend and
        # returned by the API, which use camelCase everywhere.
        return {
            "generator": self.model,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidanceScale": self.guidance_scale,
            "size": IMAGE_SIZE,
            # JSON object keys must be strings, so the int seeds are converted.
            "rejectedSeeds": {str(k): v for k, v in self.rejected_seeds.items()},
        }


class AvatarGenerator:
    """
    Lazy-loading SD 1.5 text-to-image with cached failure.

    *Lazy loading*: constructing this object loads nothing. The weights are
    read on the first ``generate`` call and dropped by ``release``, so the
    3 GB model is resident only while it is in use (golden rule 5).

    *Cached failure*: if loading fails once, the error is kept and raised
    again straight away on later calls, instead of retrying a slow load that
    will fail the same way.
    """

    def __init__(self, repo_id: str = AVATAR_DIFFUSION_REPO, device: Optional[str] = None) -> None:
        self.repo_id = repo_id
        # None means "decide when loading": CUDA if there is a GPU, else CPU.
        self.device = device
        self._pipe = None
        self._load_error: Optional[str] = None
        # Lets gpu_utils unload this pipeline when another heavy model needs
        # the memory. The same name is passed as ``keep`` below so that making
        # room for this model never unloads this model.
        gpu_utils.register_releaser("avatar-diffusion", self.release)

    def _load(self):
        """
        Return the loaded pipeline, loading it on first use.

        Raises ``AvatarGeneratorUnavailable`` (with the fetch command) when the
        weights cannot be loaded, and lets ``gpu_utils.InsufficientVRAM``
        through unchanged when the GPU is too full.
        """
        if self._pipe is not None:
            return self._pipe
        if self._load_error is not None:
            raise AvatarGeneratorUnavailable(self._load_error)

        # Imported here, not at the top: torch is slow to import, so the
        # cost is paid only when a face is actually generated.
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Make room before loading: unload other registered models if free
        # memory is short, or raise with the numbers if that is not enough.
        if device == "cpu":
            gpu_utils.ensure_host_memory(RAM_MB, "Stable Diffusion 1.5", keep="avatar-diffusion")
        if device == "cuda":
            gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "Stable Diffusion 1.5", keep="avatar-diffusion")
        try:
            from diffusers import StableDiffusionPipeline

            # fp16 (half precision) stores each number in 2 bytes instead of 4,
            # halving memory and speeding up a GPU. Many CPU operations have no
            # fp16 version, so on the CPU the weights are upcast to fp32.
            half = device == "cuda"
            # Any: diffusers' stubs type from_pretrained as a union including a
            # dummy stand-in class, which hides every real pipeline method.
            pipe: Any = StableDiffusionPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=torch.float16 if half else torch.float32,
                # Which files to read, not which precision to run in:
                # scripts/fetch_vision_models.py downloads only the
                # ``*.fp16.safetensors`` files, so this is needed on CPU too.
                variant="fp16",
                # safetensors is a weight format that cannot run code when
                # loaded, unlike pickled ``.bin`` files.
                use_safetensors=True,
                # Never reach for the network at generation time: a missing
                # file must fail here, with the fetch command, not download.
                local_files_only=True,
            )
            # A progress bar per call would only clutter the server log.
            pipe.set_progress_bar_config(disable=True)
            # Attention slicing computes attention in pieces instead of all at
            # once: a little slower, but a lower memory peak.
            pipe.enable_attention_slicing()
            self._pipe = pipe.to(device)
            self.device = device
            logger.info("Loaded %s on %s", self.repo_id, device)
            return self._pipe
        except gpu_utils.InsufficientVRAM:
            # Not cached as a load error: a full GPU is temporary, and the
            # next call may find the memory free.
            raise
        except Exception as exc:  # noqa: BLE001
            # Broad on purpose: diffusers raises many types for missing or
            # broken files. Every one becomes one clear error with the fix
            # (golden rule 7), cached so later calls fail fast.
            self._load_error = (
                f"Could not load {self.repo_id}: {exc}. Fetch the weights with: "
                f"{VISION_FETCH_COMMAND} --only avatar-diffusion"
            )
            # ``from exc`` keeps the original traceback attached for debugging.
            raise AvatarGeneratorUnavailable(self._load_error) from exc

    def generate(
        self,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE,
        seed: int = 0,
        steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> np.ndarray:
        """
        One 512x512 RGB image for ``seed``.

        Returns a ``(512, 512, 3)`` uint8 array. Raises
        ``AvatarGenerationFailed`` when the safety checker blanks the image,
        and ``AvatarGeneratorUnavailable`` when the model cannot load.
        """
        import torch

        pipe = self._load()
        # The noise generator is created on the CPU even when the model runs on
        # CUDA, so a seed gives the same starting noise whichever device the
        # pipeline is on. ``manual_seed`` returns the generator itself.
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
        # SD 1.5 ships a safety checker that replaces a flagged image with a
        # black one and sets this list. A black square is not a face, so it is
        # reported as a failure for this seed and the next seed is tried.
        # getattr with a default: the attribute is absent when no checker runs.
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged and flagged[0]:
            raise AvatarGenerationFailed(
                f"seed {seed}: the safety checker blanked the image"
            )
        # The pipeline returns PIL images. Convert to the uint8 RGB array the
        # rest of the vision code uses; ascontiguousarray gives a plain
        # row-major buffer that OpenCV and MediaPipe accept.
        return np.ascontiguousarray(np.asarray(result.images[0].convert("RGB"), dtype=np.uint8))

    def release(self) -> None:
        """Drop the pipeline so another heavy model can use the GPU."""
        if self._pipe is not None:
            # Dropping the only reference is what frees the weights;
            # empty_cache then hands the memory back to the driver.
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

    ``check`` is passed in rather than imported so tests can supply a fake
    and never load MediaPipe. A ``generator`` passed in is not released here;
    its owner decides when.

    Raises ``AvatarGenerationFailed`` listing every seed and its reason when
    no attempt passes.
    """
    # Only release a generator this call created; a caller's one is theirs.
    own = generator is None
    generator = generator or AvatarGenerator()
    rejected: Dict[int, str] = {}
    tried: List[int] = []
    # try/finally guarantees the release below runs on success, on failure
    # and on any unexpected exception.
    try:
        # max(1, ...): at least one attempt even if a caller passes 0.
        for offset in range(max(1, int(attempts))):
            # Consecutive seeds, so a rejected face is easy to reproduce from
            # the starting seed and its position.
            current = int(seed) + offset
            tried.append(current)
            try:
                image = generator.generate(prompt, negative_prompt, current, steps, guidance_scale)
            except AvatarGenerationFailed as err:
                # A blanked image fails this seed only; try the next one.
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
            # Record why this seed failed, both for the final error and for
            # the lineage of the face that eventually passes.
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


def generate_registered_avatar(
    store: "AvatarStore",
    check: Callable[[np.ndarray], "FaceQualityReport"],
    avatar_id: str,
    prompt: str = DEFAULT_PROMPT,
    seed: int = 0,
    attempts: int = 6,
    steps: int = 30,
    overwrite: bool = False,
    generator: Optional[AvatarGenerator] = None,
) -> "GeneratedAvatar":
    """
    Generate a face that passes the quality gate and register it as synthetic.

    The id is checked *before* the slow generation, so a clash costs nothing.
    Returns the generated avatar; the stored record is ``store.get(avatar_id)``.

    Raises ``AvatarError`` when the id is taken and ``overwrite`` is False,
    and whatever ``generate_avatar`` raises when no face passes.
    """
    # Imported inside the function for the same reason as the TYPE_CHECKING
    # block at the top: avatar_store imports face_engine, and with it MediaPipe.
    import provenance
    from avatar_store import AvatarError, AvatarNotFound

    if not overwrite:
        # try/except/else: ``else`` runs only when get() succeeded, which here
        # means the id already exists.
        try:
            store.get(avatar_id)
        except AvatarNotFound:
            pass
        else:
            raise AvatarError(f"avatar {avatar_id!r} already exists; pass overwrite or choose another id")
    generated = generate_avatar(
        check, prompt=prompt, seed=seed, attempts=attempts, steps=steps, generator=generator
    )
    # Registered as SYNTHETIC: the provenance sidecar says no real person is
    # depicted, which is what lets this face be used without a consent record.
    # The licence is the SD 1.5 model licence, which governs its outputs.
    store.register(
        generated.image,
        avatar_id=avatar_id,
        source=provenance.SYNTHETIC,
        licence="CreativeML OpenRAIL-M (generated output)",
        notes="Generated face; depicts no real person.",
        extra=generated.lineage(),
        overwrite=overwrite,
    )
    return generated
