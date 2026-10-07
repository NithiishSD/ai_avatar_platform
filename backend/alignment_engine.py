"""
Forced Alignment Engine & Phoneme-to-Viseme Mapper.

Extracts millisecond-accurate phoneme and viseme timestamps from audio
using torchaudio MMS_FA (Meta Multilingual Forced Aligner) with an acoustic
energy fallback when running offline or in unit tests.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import soundfile as sf
import torch

from contracts import PhonemeTimestamp
from language_registry import to_iso3
from romanizer import is_ascii, romanize

logger = logging.getLogger(__name__)

# Standard 15 Oculus/Disney Viseme Definitions:
# viseme_sil, viseme_PP, viseme_FF, viseme_TH, viseme_DD, viseme_kk,
# viseme_CH, viseme_SS, viseme_nn, viseme_RR, viseme_aa, viseme_E,
# viseme_I, viseme_O, viseme_U

# ARPAbet / CMUDict / IPA / Character phoneme-to-viseme mapping table
PHONEME_TO_VISEME: dict[str, str] = {
    # Silence / Pauses
    "SIL": "viseme_sil",
    "SP": "viseme_sil",
    "<SIL>": "viseme_sil",
    "<PAD>": "viseme_sil",
    "": "viseme_sil",
    " ": "viseme_sil",
    "-": "viseme_sil",
    ".": "viseme_sil",
    ",": "viseme_sil",
    # Bilabials (P, B, M) -> Lips together
    "P": "viseme_PP",
    "B": "viseme_PP",
    "M": "viseme_PP",
    "p": "viseme_PP",
    "b": "viseme_PP",
    "m": "viseme_PP",
    # Labiodentals (F, V) -> Lower lip to upper teeth
    "F": "viseme_FF",
    "V": "viseme_FF",
    "f": "viseme_FF",
    "v": "viseme_FF",
    # Dentals (TH, DH) -> Tongue between teeth
    "TH": "viseme_TH",
    "DH": "viseme_TH",
    "θ": "viseme_TH",
    "ð": "viseme_TH",
    # Alveolars (T, D, L) -> Tongue tip to alveolar ridge
    "T": "viseme_DD",
    "D": "viseme_DD",
    "L": "viseme_DD",
    "t": "viseme_DD",
    "d": "viseme_DD",
    "l": "viseme_DD",
    # Nasal Alveolar (N, NG)
    "N": "viseme_nn",
    "NG": "viseme_nn",
    "n": "viseme_nn",
    "ŋ": "viseme_nn",
    # Fricatives (S, Z) -> Teeth together, slight opening
    "S": "viseme_SS",
    "Z": "viseme_SS",
    "s": "viseme_SS",
    "z": "viseme_SS",
    # Palato-alveolars (SH, ZH, CH, JH) -> Rounded lips with teeth close
    "SH": "viseme_CH",
    "ZH": "viseme_CH",
    "CH": "viseme_CH",
    "JH": "viseme_CH",
    "ʃ": "viseme_CH",
    "ʒ": "viseme_CH",
    "tʃ": "viseme_CH",
    "dʒ": "viseme_CH",
    # Velars (K, G) -> Tongue back raised
    "K": "viseme_kk",
    "G": "viseme_kk",
    "k": "viseme_kk",
    "g": "viseme_kk",
    # Approximants & Rhotics (R, ER, W, Y)
    "R": "viseme_RR",
    "ER": "viseme_RR",
    "AXR": "viseme_RR",
    "r": "viseme_RR",
    "ɹ": "viseme_RR",
    "W": "viseme_U",
    "w": "viseme_U",
    "Y": "viseme_I",
    "j": "viseme_I",
    "HH": "viseme_aa",
    "h": "viseme_aa",
    # Open Vowels (AA, AH, AO, AW, AY) -> Wide open mouth
    "AA": "viseme_aa",
    "AH": "viseme_aa",
    "AO": "viseme_aa",
    "AW": "viseme_aa",
    "AY": "viseme_aa",
    "a": "viseme_aa",
    "ɑ": "viseme_aa",
    "ʌ": "viseme_aa",
    "ɔ": "viseme_aa",
    # Front Mid Vowels (AE, EH, EY) -> Moderate open, spread lips
    "AE": "viseme_E",
    "EH": "viseme_E",
    "EY": "viseme_E",
    "e": "viseme_E",
    "ɛ": "viseme_E",
    "æ": "viseme_E",
    # Front High Vowels (IH, IY) -> Narrow mouth, spread lips
    "IH": "viseme_I",
    "IY": "viseme_I",
    "i": "viseme_I",
    "ɪ": "viseme_I",
    # Back Mid/Rounded Vowels (OW, OY) -> Rounded lips, medium opening
    "OW": "viseme_O",
    "OY": "viseme_O",
    "o": "viseme_O",
    "oʊ": "viseme_O",
    # Back High/Rounded Vowels (UH, UW) -> Tightly rounded lips
    "UH": "viseme_U",
    "UW": "viseme_U",
    "u": "viseme_U",
    "ʊ": "viseme_U",
}

# Simple English Grapheme-to-Phoneme heuristic for text breakdown
CHAR_TO_PHONEME: dict[str, str] = {
    "a": "AA", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G",
    "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N",
    "o": "OW", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "u": "AH",
    "v": "V", "w": "W", "x": "S", "y": "Y", "z": "Z",
}

# Two-letter graphemes that are a single phoneme. Consumed before the
# single-character table so "ship" is SH-IH-P rather than S-HH-IH-P.
DIGRAPH_TO_PHONEME: dict[str, str] = {
    "th": "TH", "sh": "SH", "ch": "CH", "ph": "F", "ng": "NG",
    "wh": "W", "ck": "K", "gh": "G",
    "ee": "IY", "ea": "IY", "oo": "UW", "ou": "AW",
    "ow": "OW", "ai": "EY", "ay": "EY", "oa": "OW",
    "oi": "OY", "oy": "OY",
}

# Consonants after which a word-final "e" is silent ("time", "made", "name").
_SILENT_E_BLOCKERS = set("aeiouy")


def graphemes_to_phonemes(word: str) -> List[Tuple[str, int]]:
    """
    Split a lowercase word into ``(phoneme, characters_consumed)`` pairs.

    The character counts matter as much as the phonemes: the forced aligner
    gets one CTC span per character, so it needs to know how many spans to
    merge into each phoneme. ``sum(count for _, count in result) == len(word)``
    always holds, so spans and phonemes never drift apart.
    """
    low = word.lower()
    groups: List[Tuple[str, int]] = []
    index = 0
    while index < len(low):
        pair = low[index : index + 2]
        if len(pair) == 2 and pair in DIGRAPH_TO_PHONEME:
            groups.append((DIGRAPH_TO_PHONEME[pair], 2))
            index += 2
            continue

        char = low[index]
        is_final_silent_e = (
            char == "e"
            and index == len(low) - 1
            and index >= 2
            and low[index - 1] not in _SILENT_E_BLOCKERS
        )
        if is_final_silent_e and groups:
            # Silent "e" has no sound of its own; fold its span into the
            # phoneme before it so the mouth stays open through the vowel.
            phoneme, count = groups[-1]
            groups[-1] = (phoneme, count + 1)
            index += 1
            continue

        groups.append((CHAR_TO_PHONEME.get(char, "AA"), 1))
        index += 1
    return groups


class PhonemeToVisemeMapper:
    """Translates ARPAbet, IPA, or romanized phonemes to standard 15 facial visemes."""

    @staticmethod
    def map_phoneme(phoneme: str) -> str:
        """Map a single phoneme string to its corresponding viseme."""
        if not phoneme:
            return "viseme_sil"
        # Strip trailing stress digits (e.g., 'AA1' -> 'AA', 'EH0' -> 'EH')
        clean = re.sub(r"\d+$", "", phoneme.strip()).upper()
        return PHONEME_TO_VISEME.get(clean, PHONEME_TO_VISEME.get(phoneme.strip(), "viseme_aa"))

    @staticmethod
    def get_supported_visemes() -> list[str]:
        """Returns the list of 15 canonical facial visemes."""
        return [
            "viseme_sil",
            "viseme_PP",
            "viseme_FF",
            "viseme_TH",
            "viseme_DD",
            "viseme_kk",
            "viseme_CH",
            "viseme_SS",
            "viseme_nn",
            "viseme_RR",
            "viseme_aa",
            "viseme_E",
            "viseme_I",
            "viseme_O",
            "viseme_U",
        ]


class ForcedAligner:
    """
    Multilingual forced aligner for speech audio.
    
    Extracts millisecond-accurate phoneme and viseme timestamps aligned with
    an audio file and transcript text.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._mms_aligner = None
        self._mms_tokenizer = None
        self._mms_failed = False
        # How the most recent ``align`` call produced its timestamps. The
        # acoustic fallback spreads phonemes evenly by text, which is a guess:
        # callers surface this so a guessed timeline is never mistaken for a
        # measured one (golden rule 1).
        self.last_method: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None

    def _get_mms_pipeline(self):
        """Lazy loader for torchaudio MMS_FA pipeline."""
        if self._mms_aligner is not None:
            return self._mms_aligner, self._mms_tokenizer
        if self._mms_failed:
            return None, None

        try:
            import torchaudio.pipelines as pipelines
            bundle = pipelines.MMS_FA
            tokenizer = bundle.get_tokenizer()
            model = bundle.get_model()
            try:
                model = model.to(self.device)
            except RuntimeError as exc:
                if self.device == "cpu":
                    raise
                # Usually another process holding the GPU. The aligner is
                # small enough to run on CPU, and a slower real alignment
                # beats a fast guess.
                logger.warning(
                    "MMS_FA could not be placed on %s (%s); running the aligner on CPU instead.",
                    self.device, str(exc).splitlines()[0],
                )
                torch.cuda.empty_cache()
                self.device = "cpu"
                model = model.to("cpu")
            self._mms_aligner = model
            self._mms_tokenizer = tokenizer
            logger.info("Loaded torchaudio MMS_FA forced aligner on %s", self.device)
            return self._mms_aligner, self._mms_tokenizer
        except Exception as exc:
            self.last_fallback_reason = f"MMS_FA could not be loaded: {str(exc).splitlines()[0]}"
            logger.warning("Could not load MMS_FA pipeline (%s). Using acoustic fallback.", exc)
            self._mms_failed = True
            return None, None

    def align(
        self,
        audio_path_or_tensor: Union[str, Path, torch.Tensor],
        transcript: str,
        sample_rate: int = 24000,
        language: str = "en",
    ) -> List[PhonemeTimestamp]:
        """
        Align transcript text against audio to generate PhonemeTimestamp objects.

        Args:
            audio_path_or_tensor: File path to audio file or 1D/2D PyTorch Tensor.
            transcript: The spoken text string to align.
            sample_rate: Sample rate of the audio (default 24000).
            language: Language code for alignment (e.g. 'en', 'es', 'fr').

        Returns:
            List of validated PhonemeTimestamp objects matching AvatarRenderJob contracts.
        """
        # Load audio and determine duration
        duration_ms = 0
        if isinstance(audio_path_or_tensor, (str, Path)):
            path_str = str(audio_path_or_tensor)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Audio file not found: {path_str}")
            data, sr = sf.read(path_str)
            sample_rate = sr
            num_samples = len(data) if data.ndim == 1 else len(data[:, 0])
            duration_ms = int((num_samples / sample_rate) * 1000)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                waveform = waveform.T
        elif isinstance(audio_path_or_tensor, torch.Tensor):
            waveform = audio_path_or_tensor.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            num_samples = waveform.shape[-1]
            duration_ms = int((num_samples / sample_rate) * 1000)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_path_or_tensor)}")

        if duration_ms <= 0:
            duration_ms = 1000  # Minimum fallback

        # Clean transcript text
        cleaned_text = transcript.strip() if transcript else ""
        if not cleaned_text:
            # Silence timestamp if text is empty
            self.last_method = "silence"
            return [
                PhonemeTimestamp(
                    phoneme="SIL",
                    viseme="viseme_sil",
                    startMs=0,
                    endMs=max(duration_ms, 50),
                )
            ]

        # Both aligners work over a Latin alphabet, so non-Latin script has to
        # be transliterated before either one sees it.
        alignable_text = self._prepare_for_alignment(cleaned_text, language)

        # Try neural MMS_FA alignment if possible
        model, tokenizer = self._get_mms_pipeline()
        if model is not None and tokenizer is not None:
            for attempt in range(2):
                try:
                    timestamps = self._align_mms(waveform, sample_rate, alignable_text, model, tokenizer, duration_ms)
                    if timestamps:
                        self.last_method = "mms_fa"
                        self.last_fallback_reason = None
                        return timestamps
                    self.last_fallback_reason = "MMS_FA returned no spans for this transcript"
                    break
                except Exception as err:  # noqa: BLE001
                    out_of_memory = "out of memory" in str(err).lower()
                    if attempt == 0 and self.device != "cpu" and out_of_memory:
                        logger.warning("MMS_FA ran out of GPU memory; retrying the alignment on CPU.")
                        torch.cuda.empty_cache()
                        self.device = "cpu"
                        model = self._mms_aligner = model.to("cpu")
                        continue
                    self.last_fallback_reason = f"MMS_FA inference failed: {str(err).splitlines()[0]}"
                    break

        # Fallback to acoustic / syllabic aligner. Loud on purpose: these
        # timestamps are spread by text length, not measured from the audio.
        self.last_method = "acoustic-fallback"
        logger.warning(
            "FORCED ALIGNMENT FELL BACK to the acoustic guess (%s). Phoneme timing is "
            "estimated, not measured; lip sync will be loose.",
            self.last_fallback_reason or "MMS_FA unavailable",
        )
        return self._acoustic_align(alignable_text, duration_ms)

    @staticmethod
    def _prepare_for_alignment(transcript: str, language: str) -> str:
        """
        Transliterate a non-Latin transcript so the aligners can consume it.

        MMS_FA's dictionary is a-z plus apostrophe and the acoustic fallback
        splits Latin graphemes, so Devanagari, Tamil, Cyrillic or Han input
        would otherwise survive as zero alignable words and silently produce
        timings unrelated to the speech. Romanizing first keeps the CTC frame
        spans as the timing evidence.

        The returned phonemes are derived from the romanization rather than
        from the original orthography. That is an approximation of the spoken
        sounds, but it drives the correct mouth shapes, which is what the
        viseme contract needs.
        """
        if not transcript or is_ascii(transcript):
            return transcript

        romanized = romanize(transcript, lcode=to_iso3(language))
        if romanized and romanized.strip():
            logger.info(
                "Romanized %s transcript for alignment (%d chars -> %d chars)",
                language,
                len(transcript),
                len(romanized),
            )
            return romanized.strip()

        logger.warning(
            "Cannot align non-Latin %s transcript: uroman is not installed, so "
            "timings will be acoustic estimates rather than forced alignment. "
            "Install it with `pip install uroman`.",
            language,
        )
        return transcript

    # Minimum audible span for one phoneme; shorter CTC spans get widened.
    MIN_PHONEME_MS = 20
    # A gap this long between two words is rendered as an explicit closed mouth.
    SILENCE_GAP_MS = 60

    @staticmethod
    def _alignable_words(transcript: str) -> List[str]:
        """
        Reduce a transcript to the words MMS_FA can actually align.

        The MMS_FA dictionary holds a-z plus apostrophe, so digits, punctuation
        and non-Latin characters are stripped; anything left empty is dropped.
        """
        words: List[str] = []
        for raw in re.findall(r"[A-Za-z']+", transcript):
            cleaned = raw.lower().strip("'")
            if cleaned:
                words.append(cleaned)
        return words

    def _align_mms(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        transcript: str,
        model: torch.nn.Module,
        tokenizer: any,
        duration_ms: int,
    ) -> List[PhonemeTimestamp]:
        """
        Align with torchaudio MMS_FA and keep the CTC frame spans.

        MMS_FA is a character-level CTC aligner: ``forced_align`` assigns every
        input frame to a target character, and ``merge_tokens`` collapses that
        into one frame span per character. Those spans are the actual timing
        evidence, so phonemes are built by merging the spans of the characters
        that form them rather than by dividing the duration evenly.
        """
        import torchaudio.functional as F

        # MMS_FA expects 16 kHz mono.
        audio_16k = waveform
        if audio_16k.ndim == 2 and audio_16k.shape[0] > 1:
            audio_16k = audio_16k.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio_16k = F.resample(audio_16k, orig_freq=sample_rate, new_freq=16000)
        audio_16k = audio_16k.to(self.device)

        words = self._alignable_words(transcript)
        if not words:
            return self._acoustic_align(transcript, duration_ms)

        token_lists = tokenizer(words)
        pairs = [(w, t) for w, t in zip(words, token_lists) if t and len(t) == len(w)]
        if not pairs:
            raise RuntimeError("no transcript words survived MMS_FA tokenization")
        words = [w for w, _ in pairs]
        token_lists = [t for _, t in pairs]

        with torch.inference_mode():
            emission, _ = model(audio_16k)
        emission = emission.cpu()

        num_frames = int(emission.shape[1])
        targets = torch.tensor(
            [token for tokens in token_lists for token in tokens], dtype=torch.int32
        ).unsqueeze(0)
        if targets.shape[-1] == 0 or targets.shape[-1] > num_frames:
            raise RuntimeError(
                f"transcript has {targets.shape[-1]} tokens but the audio only "
                f"yields {num_frames} CTC frames - audio and text do not match"
            )

        aligned_tokens, scores = F.forced_align(emission, targets, blank=0)
        spans = F.merge_tokens(aligned_tokens[0], scores[0].exp())
        if len(spans) != targets.shape[-1]:
            raise RuntimeError(
                f"CTC produced {len(spans)} spans for {targets.shape[-1]} tokens"
            )

        # One emission frame covers this many milliseconds of the source audio.
        ms_per_frame = duration_ms / max(num_frames, 1)

        result: List[PhonemeTimestamp] = []
        span_index = 0
        previous_end = 0

        for word in words:
            groups = graphemes_to_phonemes(word)
            word_spans = spans[span_index : span_index + len(word)]
            span_index += len(word)

            word_start = int(round(word_spans[0].start * ms_per_frame))
            if word_start - previous_end >= self.SILENCE_GAP_MS:
                result.append(
                    PhonemeTimestamp(
                        phoneme="SIL",
                        viseme="viseme_sil",
                        startMs=previous_end,
                        endMs=word_start,
                    )
                )

            offset = 0
            for phoneme, char_count in groups:
                group_spans = word_spans[offset : offset + char_count]
                offset += char_count
                if not group_spans:
                    continue
                start_ms = int(round(group_spans[0].start * ms_per_frame))
                end_ms = int(round(group_spans[-1].end * ms_per_frame))
                start_ms = max(start_ms, previous_end)
                end_ms = max(end_ms, start_ms + self.MIN_PHONEME_MS)
                result.append(
                    PhonemeTimestamp(
                        phoneme=phoneme,
                        viseme=PhonemeToVisemeMapper.map_phoneme(phoneme),
                        startMs=start_ms,
                        endMs=end_ms,
                    )
                )
                previous_end = end_ms

        if not result:
            raise RuntimeError("MMS_FA alignment produced no phoneme spans")

        return self._normalize_timestamps(result, duration_ms)

    def _acoustic_align(self, transcript: str, duration_ms: int) -> List[PhonemeTimestamp]:
        """
        Acoustic heuristic forced-aligner for offline mode, testing, and edge cases.
        Decomposes words into syllables and phonemes with natural speech rhythm.
        """
        tokens = re.findall(r"\w+|[^\w\s]", transcript)
        if not tokens:
            return [
                PhonemeTimestamp(
                    phoneme="SIL",
                    viseme="viseme_sil",
                    startMs=0,
                    endMs=max(duration_ms, 50),
                )
            ]

        # Break tokens down into phoneme units
        phoneme_units: list[str] = []
        for tok in tokens:
            if re.match(r"^\w+$", tok):
                # Word: same grapheme-to-phoneme split the CTC path uses, so
                # both aligners emit the same phoneme sequence for a sentence.
                phoneme_units.extend(p for p, _ in graphemes_to_phonemes(tok))
                phoneme_units.append("SP")  # Word boundary pause
            elif tok in (".", "!", "?", ";"):
                phoneme_units.append("SIL")  # Sentence boundary pause
            elif tok == ",":
                phoneme_units.append("SP")

        # Trim trailing silences
        while phoneme_units and phoneme_units[-1] in ("SP", "SIL"):
            phoneme_units.pop()

        if not phoneme_units:
            return [
                PhonemeTimestamp(
                    phoneme="SIL",
                    viseme="viseme_sil",
                    startMs=0,
                    endMs=max(duration_ms, 50),
                )
            ]

        # Allocate time slices proportionally
        # Vowels get ~1.5x weight, consonants 1.0x, short pauses 0.5x, silences 1.0x
        weights = []
        for p in phoneme_units:
            vis = PhonemeToVisemeMapper.map_phoneme(p)
            if vis in ("viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U"):
                weights.append(1.5)
            elif p in ("SP", "SIL"):
                weights.append(0.6)
            else:
                weights.append(1.0)

        total_weight = sum(weights)
        raw_timestamps: List[PhonemeTimestamp] = []
        curr_ms = 0

        for p, w in zip(phoneme_units, weights):
            unit_duration = int(math.floor((w / total_weight) * duration_ms))
            unit_duration = max(unit_duration, 25)  # At least 25ms per phoneme
            end_ms = curr_ms + unit_duration
            if end_ms > duration_ms:
                end_ms = duration_ms
            if end_ms <= curr_ms:
                end_ms = curr_ms + 25

            raw_timestamps.append(
                PhonemeTimestamp(
                    phoneme=p,
                    viseme=PhonemeToVisemeMapper.map_phoneme(p),
                    startMs=curr_ms,
                    endMs=end_ms,
                )
            )
            curr_ms = end_ms

        return self._normalize_timestamps(raw_timestamps, duration_ms)

    def _normalize_timestamps(
        self,
        timestamps: List[PhonemeTimestamp],
        total_duration_ms: int,
    ) -> List[PhonemeTimestamp]:
        """Ensure all timestamps are strictly valid, sequential, and monotonic."""
        if not timestamps:
            return [
                PhonemeTimestamp(
                    phoneme="SIL",
                    viseme="viseme_sil",
                    startMs=0,
                    endMs=max(total_duration_ms, 50),
                )
            ]

        normalized: List[PhonemeTimestamp] = []
        prev_end = 0

        for item in timestamps:
            start = max(item.start_ms, prev_end)
            end = max(item.end_ms, start + 20)
            if start >= total_duration_ms:
                break
            if end > total_duration_ms:
                end = total_duration_ms

            if end > start:
                normalized.append(
                    PhonemeTimestamp(
                        phoneme=item.phoneme,
                        viseme=item.viseme,
                        startMs=start,
                        endMs=end,
                    )
                )
                prev_end = end

        # Pad remaining tail up to total_duration_ms if gap exists
        if normalized and normalized[-1].end_ms < total_duration_ms - 20:
            last_end = normalized[-1].end_ms
            normalized.append(
                PhonemeTimestamp(
                    phoneme="SIL",
                    viseme="viseme_sil",
                    startMs=last_end,
                    endMs=total_duration_ms,
                )
            )

        return normalized
