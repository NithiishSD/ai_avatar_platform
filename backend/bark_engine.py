"""
Bark (small) for two-speaker dialogue: text with [S1]/[S2] tags -> one clip.

Dialogue mode used to route to Dia-1.6B, which cannot run on this stack (see
model_registry.UNRUNNABLE). Bark is a generative text-to-audio model from Suno,
MIT licensed, supported natively by the pinned transformers (BarkModel), and
its small variant (1.7 GB) fits the 6 GB card. It speaks in any of its
*voice presets* - short prompts that fix a speaker's timbre - so a dialogue is
rendered by giving each tagged speaker its own preset and joining the turns.

Bark samples its output (it is a GPT-style model generating audio tokens), so
every generation is seeded; the seed is scoped to the call so it cannot
disturb any other model's random state.

**How to say this in an interview:** "Multi-speaker dialogue is assembled
turn by turn: the script is split on speaker tags, each turn is generated
with that speaker's voice preset, and the turns are concatenated with a
short pause - deterministic because generation is seeded."

**How Bark makes sound, in three stages.** Bark never predicts a waveform
directly. It works on *tokens*: small integers, each one an entry in a learned
codebook. Stage one turns text into *semantic tokens* (what is being said).
Stage two turns those into *coarse acoustic tokens* (the first codebooks of
the EnCodec audio codec: rough timbre and prosody). Stage three adds the
*fine* codebooks (detail). EnCodec then decodes all of them into a waveform.
A voice preset is a short recorded prompt for each of the three stages,
which is why a preset is three arrays (see ``PRESET_PARTS``).

Where this sits: ``voice_engine.VoiceEngineRouter`` sends dialogue-mode
requests here; the WAV written by ``synthesize_dialogue`` then follows the
same path as every other engine's output (watermark, then alignment).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gpu_utils

logger = logging.getLogger(__name__)

# The Hugging Face repository id; the snapshot must already be on disk.
BARK_REPO = "suno/bark-small"
# Measured 8 Oct 2026: adds 1840 MiB to the process peak on a CPU host.
RAM_MB = 1900
# Files the model and the two dialogue presets need; the registry checks these.
BARK_FILES = ("config.json", "pytorch_model.bin")
# One preset per speaker tag. A male and a female English voice, so the two
# speakers are easy to tell apart.
DIALOGUE_VOICES = {1: "v2/en_speaker_6", 2: "v2/en_speaker_9"}
# The three arrays inside every preset, one per generation stage (see the
# module docstring). Each is stored as its own .npy file in the snapshot.
PRESET_PARTS = ("semantic_prompt", "coarse_prompt", "fine_prompt")
# Silence inserted between two turns, so speakers do not talk over each other.
TURN_GAP_SECONDS = 0.3
# Bark is trained on clips of about 13 s; longer turns are generated sentence
# by sentence so none runs past what the model can say in one go.
MAX_CHARS_PER_GENERATION = 220

# Matches a speaker tag such as [S1] or [s2]; group 1 captures the number.
# Only speakers 1 and 2 exist, matching the two entries in DIALOGUE_VOICES.
_TAG = re.compile(r"\[(?:S|s)([12])\]")


class BarkUnavailable(RuntimeError):
    """Bark or one of its presets cannot be loaded; the message names the fix."""


def split_dialogue(text: str) -> List[Tuple[int, str]]:
    """
    ``"[S1] Hi. [S2] Hello."`` -> ``[(1, "Hi."), (2, "Hello.")]``.

    Text before the first tag belongs to speaker 1, and untagged text is one
    speaker-1 turn, so plain text still works in dialogue mode.
    """
    turns: List[Tuple[int, str]] = []
    # ``position`` is where the text after the previous tag starts; the text
    # between it and the next tag belongs to the speaker that tag introduced.
    speaker, position = 1, 0
    for match in _TAG.finditer(text):
        chunk = text[position:match.start()].strip()
        # Empty chunks (two tags in a row, or a tag at the start) add no turn.
        if chunk:
            turns.append((speaker, chunk))
        speaker, position = int(match.group(1)), match.end()
    # Whatever follows the last tag is the final turn.
    tail = text[position:].strip()
    if tail:
        turns.append((speaker, tail))
    return turns


def chunk_turn(turn: str, limit: int = MAX_CHARS_PER_GENERATION) -> List[str]:
    """Split a long turn at sentence ends so each piece fits one generation."""
    if len(turn) <= limit:
        return [turn]
    # The lookbehind splits on the whitespace after . ! or ? and keeps the
    # punctuation on the sentence, so each piece still ends naturally.
    sentences = re.split(r"(?<=[.!?])\s+", turn)
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        # Greedy packing: add sentences to the current piece until the next
        # one would push it past the limit (+1 for the joining space). A
        # single sentence longer than the limit still becomes its own piece.
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


class BarkEngine:
    """
    Lazily loaded Bark model with presets read from the local snapshot.

    *Lazy loading* means the constructor only records settings; the 1.7 GB
    of weights are read on the first generation. The server can therefore
    start without Bark and only pay for it when dialogue mode is used, and
    ``release`` can drop it so another heavy model gets the memory.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Record the device; nothing is loaded until the first generation."""
        # None means "decide at load time": CUDA when visible, else CPU.
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._snapshot: Optional[Path] = None
        # The first load error is remembered so later requests fail fast with
        # the same fix message instead of retrying a broken install each time.
        self._failure: Optional[str] = None
        # Preset name -> its three arrays, read once from disk.
        self._presets: Dict[str, Dict[str, np.ndarray]] = {}

    def _load(self) -> Tuple[Any, Any]:
        """Return ``(model, processor)``, loading them on first use."""
        if self._model is not None:
            return self._model, self._processor
        if self._failure is not None:
            raise BarkUnavailable(self._failure)
        # Before the try: running short of RAM is not a broken install, so it
        # must not be cached as a permanent load failure.
        if self.device in (None, "cpu"):
            gpu_utils.ensure_host_memory(RAM_MB, "Bark")
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor, BarkModel

            # local_files_only: never download at request time.
            snapshot = Path(snapshot_download(BARK_REPO, local_files_only=True))
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
            # transformers types its lazily imported model classes as possibly
            # None; at runtime this is always the class (same as D-21).
            model = BarkModel.from_pretrained(  # pyrefly: ignore[not-callable]
                str(snapshot), local_files_only=True
            ).to(device).eval()
            self._model, self._processor, self._snapshot, self.device = model, processor, snapshot, device
            logger.info("Loaded Bark (small) on %s", device)
            return model, processor
        except Exception as exc:  # noqa: BLE001 - cached and raised with the fix
            self._failure = (
                f"Bark could not be loaded ({str(exc).splitlines()[0]}). "
                "Fetch it with: python scripts/fetch_models.py --only bark"
            )
            logger.error(self._failure)
            raise BarkUnavailable(self._failure) from exc

    def _preset(self, name: str) -> Dict[str, np.ndarray]:
        """
        A voice preset as arrays, read from disk.

        Passing the arrays (rather than the preset's name) keeps the
        processor from looking the preset up on the Hub at request time.
        """
        if name not in self._presets:
            # _load runs before any preset is read, so the snapshot is known.
            assert self._snapshot is not None
            # "v2/en_speaker_6" lives in speaker_embeddings/v2/ as
            # en_speaker_6_semantic_prompt.npy and so on.
            folder = self._snapshot / "speaker_embeddings" / Path(name).parent
            stem = Path(name).name
            try:
                self._presets[name] = {
                    part: np.load(folder / f"{stem}_{part}.npy") for part in PRESET_PARTS
                }
            except FileNotFoundError as exc:
                raise BarkUnavailable(
                    f"Bark voice preset {name!r} is not on disk ({exc.filename}). "
                    "Fetch it with: python scripts/fetch_models.py --only bark"
                ) from exc
        return self._presets[name]

    def _generate(self, text: str, voice: str, seed: int) -> Tuple[np.ndarray, int]:
        """
        One text piece in one voice -> ``(waveform, sample_rate)``.

        ``model.generate`` runs all three token stages and the EnCodec decoder
        internally; the preset conditions each stage on the speaker.
        """
        import torch

        model, processor = self._load()
        # The processor tokenizes the text and attaches the preset arrays.
        inputs = processor(text, voice_preset=self._preset(voice))
        # Move tensors to the model's device; non-tensor entries pass through.
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # fork_rng must also save and restore the CUDA generator when the
        # model runs on the GPU; on CPU there is no CUDA state to protect.
        devices = [torch.cuda.current_device()] if str(self.device).startswith("cuda") else []
        # fork_rng scopes the seed to this generation (see module docstring).
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            audio = model.generate(**inputs)
        # The sample rate comes from the model's own config rather than a constant.
        rate = int(model.generation_config.sample_rate)
        return audio.squeeze().float().cpu().numpy(), rate

    def synthesize_dialogue(self, text: str, output_path: Path, seed: int = 0) -> Tuple[int, float, int]:
        """
        Render a tagged dialogue to ``output_path``.

        Returns ``(sample_rate, duration_seconds, turn_count)``.
        """
        import soundfile as sf

        turns = split_dialogue(text)
        if not turns:
            raise ValueError("dialogue text has no words after removing the [S1]/[S2] tags")
        pieces: List[np.ndarray] = []
        # Starting value for the first gap; it is replaced by the rate the
        # model reports after the first generation.
        rate = 24000
        for index, (speaker, turn) in enumerate(turns):
            # A pause before every turn except the first.
            if pieces:
                pieces.append(np.zeros(int(TURN_GAP_SECONDS * rate), dtype=np.float32))
            for offset, chunk in enumerate(chunk_turn(turn)):
                # A different seed per piece: the same seed for every piece
                # would make two identical sentences sound like a loop.
                audio, rate = self._generate(chunk, DIALOGUE_VOICES[speaker], seed + 97 * index + offset)
                pieces.append(audio.astype(np.float32))
        # Join every turn and gap into one continuous clip.
        audio = np.concatenate(pieces)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak  # Bark can overshoot; clipping would distort
        sf.write(output_path, audio, rate)
        return rate, len(audio) / rate, len(turns)

    def release(self) -> None:
        """Drop the model and presets; they reload on next use."""
        self._model = None
        self._processor = None
        self._presets.clear()
