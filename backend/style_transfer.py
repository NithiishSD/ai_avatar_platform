"""
Style transfer for an avatar (R-43, T8.6): the same face as a cartoon, a painting, a sketch.

Stable Diffusion 1.5 (the weights the avatar generator already uses) in image-to-image mode: the
registered photo is the starting point and ``strength`` says how far the picture may move from it
(0 = unchanged, 1 = ignore the photo). Each style is a fixed prompt and strength, not free text, for
the same reason the generator takes fixed attributes: a free prompt could ask for anything.

Two things are measured and recorded, every time:

* **identity**: SFace cosine similarity between the source photo and the stylised picture
  (``visual_metrics``), next to OpenCV's same-person threshold. A stronger style moves further from
  the person; the number says how far, so nobody has to guess.
* **consent**: a stylised avatar is a derivative of the source. It inherits the source's
  provenance (a synthetic face stays synthetic; a real person's photo keeps its subject and consent
  basis) and records what it was derived from. A source that may not be used cannot be stylised.

The result must still pass the face quality gate, or it could not be animated.

Where it sits: ``POST /avatars/{id}/stylize`` in ``app.py`` calls ``stylize_registered_avatar``. The
styled picture is registered in the avatar store as a new avatar, and from then on it is animated
exactly like any other registered photo.

Concepts used below, explained once:

**Image-to-image diffusion**: a text-to-image model normally starts from pure noise. Image-to-image
instead adds noise to an existing picture and then removes it under the guidance of a prompt. How
much noise is added is ``strength``, so a low strength keeps most of the original layout and face.

**Lazy loading**: the multi-gigabyte pipeline is built on the first request, not at import, so the
server starts fast and a machine that never stylises never pays for it (golden rule 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

import gpu_utils
from avatar_generator import AVATAR_DIFFUSION_REPO, IMAGE_SIZE, RAM_MB, REQUIRED_VRAM_MB, AvatarGeneratorUnavailable
from model_registry import VISION_FETCH_COMMAND

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Style:
    """One preset: the prompt that describes the look, and how far it may move from the photo."""

    prompt: str
    strength: float  # how far img2img may move from the photo


# A negative prompt lists what the picture should *not* contain; the sampler steers away from it.
# The entries are the common failure modes of portrait generation (extra eyes, cropped heads, text).
NEGATIVE = "blurry, deformed face, extra eyes, extra limbs, text, watermark, multiple people, cropped head"
# "realistic" uses the lowest strength because it only polishes the photo; the stylised looks need
# more freedom to change texture and line, which is also why they lose more of the person's identity.
STYLES: Dict[str, Style] = {
    "realistic": Style("a high quality realistic portrait photograph of the same person, natural light, sharp focus", 0.30),
    "cartoon": Style("a cartoon portrait illustration of the same person, clean lines, flat colours, cel shading", 0.55),
    "painting": Style("an oil painting portrait of the same person, visible brush strokes, classical", 0.50),
    "sketch": Style("a pencil sketch portrait of the same person, graphite on paper, cross hatching", 0.55),
}


class StyleTransferFailed(RuntimeError):
    """The styled picture could not be used; the message says why."""


class StyleTransfer:
    """Lazy SD 1.5 image-to-image pipeline, local weights only (same rules as ``AvatarGenerator``)."""

    def __init__(self, repo_id: str = AVATAR_DIFFUSION_REPO, device: Optional[str] = None) -> None:
        self.repo_id = repo_id
        self.device = device
        # None until the first stylize() call loads it.
        self._pipe: Any = None
        # Lets gpu_utils unload this pipeline when another heavy model needs the memory.
        gpu_utils.register_releaser("style-transfer", self.release)

    def _load(self) -> Any:
        """
        Build the img2img pipeline once and return it.

        Frees memory first by evicting other registered models (RAM on CPU, VRAM on GPU), and raises
        ``AvatarGeneratorUnavailable`` with the fetch command when the weights are not on disk.
        """
        if self._pipe is not None:
            return self._pipe
        # Imported here, not at the top, so importing this module stays cheap (torch takes seconds).
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            gpu_utils.ensure_host_memory(RAM_MB, "Stable Diffusion 1.5 (img2img)", keep="style-transfer")
        else:
            gpu_utils.ensure_vram(REQUIRED_VRAM_MB, "Stable Diffusion 1.5 (img2img)", keep="style-transfer")
        try:
            from diffusers import StableDiffusionImg2ImgPipeline

            # float16 halves memory on the GPU; CPUs run float16 slowly or not at all, so they get
            # float32. local_files_only=True means a missing file fails here with the fetch command
            # instead of starting a download in the middle of a request.
            pipe: Any = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.repo_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16", use_safetensors=True, local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Any load failure becomes one typed error whose message says how to fix it (golden rule 7).
            raise AvatarGeneratorUnavailable(
                f"Could not load {self.repo_id} for style transfer: {exc}. Fetch the weights with: "
                f"{VISION_FETCH_COMMAND} --only avatar-diffusion"
            ) from exc
        # No tqdm progress bars in the server log.
        pipe.set_progress_bar_config(disable=True)
        # Attention slicing computes attention in pieces: slower, but a lower memory peak on a 6 GB card.
        pipe.enable_attention_slicing()
        self._pipe = pipe.to(device)
        self.device = device
        return self._pipe

    def stylize(self, image: np.ndarray, style: str, seed: int = 0, steps: int = 30) -> np.ndarray:
        """``image`` (RGB uint8, any size) in ``style``, as a 512x512 RGB uint8 picture."""
        import cv2
        import torch
        from PIL import Image

        chosen = STYLES[style]
        # SD 1.5 was trained at 512x512; INTER_AREA is the resampling filter that shrinks without aliasing.
        square = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        # The generator is seeded on the CPU so the same seed gives the same noise on any device, which
        # makes a result reproducible from the seed recorded in its lineage.
        result = self._load()(
            prompt=chosen.prompt, negative_prompt=NEGATIVE, image=Image.fromarray(square), strength=chosen.strength,
            num_inference_steps=int(steps), guidance_scale=7.0, generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        )
        # diffusers' safety checker replaces a flagged image with a black one; using that black image
        # would fail later in a confusing way, so it is rejected here with an actionable message.
        flagged = getattr(result, "nsfw_content_detected", None)
        if flagged and flagged[0]:
            raise StyleTransferFailed("the safety checker blanked the styled image; try another seed")
        # ascontiguousarray: a single packed block of memory, which OpenCV functions downstream expect.
        return np.ascontiguousarray(np.asarray(result.images[0].convert("RGB"), dtype=np.uint8))

    def release(self) -> None:
        """Drop the pipeline and return its memory; the next stylize() reloads it."""
        if self._pipe is not None:
            self._pipe = None
            gpu_utils.empty_cache()


# One pipeline per process: two copies of SD 1.5 would not fit in 6 GB.
_shared: Optional[StyleTransfer] = None


def shared_transfer() -> StyleTransfer:
    """The process-wide ``StyleTransfer``, created on first use (its weights load later still)."""
    global _shared
    if _shared is None:
        _shared = StyleTransfer()
    return _shared


def stylize_registered_avatar(
    store: Any,
    check: Callable[[np.ndarray], Any],
    source_id: str,
    style: str,
    new_id: str,
    seed: int = 0,
    steps: int = 30,
    transfer: Optional[StyleTransfer] = None,
    identity: Optional[Callable[[np.ndarray, np.ndarray], Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Stylise ``source_id`` and register the result as ``new_id`` with inherited provenance.

    Raises ``AvatarConsentError`` / ``AvatarNotFound`` for the source, ``StyleTransferFailed`` when
    the result fails the face quality gate. Returns what was made and how close it stays to the person.
    """
    # A deferred import: avatar_store (and the face_engine it imports) loads only when a stylise is
    # actually requested, not whenever this module is imported.
    from avatar_store import AvatarError, AvatarNotFound

    # Checked before anything expensive: a typo should not cost a model load.
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; one of {sorted(STYLES)}")
    # try/except/else: the else branch runs only when no exception was raised, which here means the
    # id is already taken. Checked up front so a minute of diffusion is not wasted on a doomed register.
    try:
        store.get(new_id)
    except AvatarNotFound:
        pass
    else:
        raise AvatarError(f"avatar {new_id!r} already exists; choose another id")
    source_image = store.load_image(source_id)  # consent is checked here
    parent = store.get(source_id).provenance
    # The shared pipeline stays loaded between requests (a reload costs a minute and, on a busy
    # 14 GB host, may not fit again); any other heavy model evicts it through gpu_utils' releasers.
    transfer = transfer or shared_transfer()
    styled = transfer.stylize(source_image, style, seed=seed, steps=steps)
    # ``check`` is the face quality gate passed in by the caller (injected so tests can fake it).
    report = check(styled)
    if not report.passed:
        raise StyleTransferFailed(
            f"the {style} picture failed the face quality gate ({'; '.join(i.code for i in report.errors)}), "
            "so it could not be animated; try another seed"
        )
    # ``identity`` is injectable for the same reason; the default is the SFace face-embedding scorer.
    # Cosine similarity of two embeddings: 1.0 is identical, lower is less alike. None means a face
    # could not be found in one of the pictures, which is reported rather than guessed.
    if identity is None:
        from visual_metrics import shared_scorer

        scorer = shared_scorer()

        def identity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
            first, second = scorer.embedding(a), scorer.embedding(b)
            return None if first is None or second is None else scorer.cosine(first, second)

    cosine = identity(source_image, styled)
    chosen = STYLES[style]
    # Everything needed to reproduce or audit the picture is stored with the new avatar.
    lineage = {"derivedFrom": source_id, "style": style, "prompt": chosen.prompt, "strength": chosen.strength, "seed": int(seed),
               "steps": int(steps), "generator": transfer.repo_id, "identityCosine": None if cosine is None else round(cosine, 3)}
    # The provenance fields are copied from the parent: a derivative carries the same consent.
    store.register(
        styled, avatar_id=new_id, source=str(parent.get("source") or ""), subject=str(parent.get("speaker") or parent.get("subject") or ""),
        licence=str(parent.get("licence") or ""), consent_basis=str(parent.get("consentBasis") or ""),
        notes=f"{style} style derived from {source_id}; inherits its provenance", extra=lineage,
    )
    from visual_metrics import SFACE_SAME_PERSON_COSINE

    return {"avatarId": new_id, "derivedFrom": source_id, "style": style, "seed": int(seed),
            "identity": {"cosine": lineage["identityCosine"],
                         "percent": None if cosine is None else round(100 * cosine, 1),
                         "sameIdentityThreshold": SFACE_SAME_PERSON_COSINE,
                         "samePerson": None if cosine is None else cosine >= SFACE_SAME_PERSON_COSINE,
                         "method": "SFace cosine, source photo vs styled picture"}}
