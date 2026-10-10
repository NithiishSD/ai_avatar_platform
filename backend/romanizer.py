"""
Shared lazy uroman romanizer.

Two subsystems need the same romanizer for different reasons:

* ``mms_engine`` — MMS-TTS checkpoints whose tokenizer sets ``is_uroman``
  expect romanized input, so synthesizing non-Latin script depends on it.
* ``alignment_engine`` — torchaudio MMS_FA is a character-level CTC aligner
  over an a-z dictionary. Without romanization a Hindi or Tamil transcript
  reduces to zero alignable words, and the aligner degrades to energy-based
  guessing that yields visemes unrelated to the speech.

Constructing ``uroman.Uroman`` loads its transliteration tables, so the
instance is built once per process and shared. ``uroman`` is an optional
dependency: every caller must handle ``None``.

*Romanization* means writing text from any script in Latin letters by sound -
"नमस्ते" becomes "namaste". It is not translation: the words are unchanged, only
the alphabet. Both consumers above are models trained on a-z, so this is the
adapter that lets a 1000-language pipeline reuse them.

**How to say this in an interview:** "Non-Latin scripts are romanized before
the character-level aligner sees them, because a CTC model over an a-z lexicon
silently matches nothing otherwise and degrades to guessed timings."
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The *cached-failure* pattern, in two variables rather than one.
# _romanizer alone could not distinguish "not tried yet" from "tried and
# unavailable", so a missing dependency would retry (and re-log) on every call.
_romanizer = None
_checked = False


def get_romanizer():
    """Return the process-wide ``uroman.Uroman``, or ``None`` if not installed."""
    global _romanizer, _checked
    # The whole point of _checked: a second call after a failed import returns
    # None immediately instead of attempting the import again.
    if _checked:
        return _romanizer
    # Set *before* the attempt, so an exception still marks it as tried.
    _checked = True
    try:
        import uroman as ur

        # This is the expensive line - it loads the transliteration tables -
        # which is why the whole module exists rather than each caller doing it.
        _romanizer = ur.Uroman()
        logger.info("uroman romanizer available")
    except Exception as exc:  # noqa: BLE001
        # logger.info, not warning or error: this is an optional dependency and
        # English synthesis is unaffected, so a missing uroman is a fact to
        # record rather than a problem to flag. The consequence is named in the
        # message so the log is self-explanatory (golden rule 7).
        logger.info(
            "uroman not available (%s); non-Latin script support is limited", exc
        )
        _romanizer = None
    return _romanizer


def as_text(result: Any) -> str:
    """
    uroman's text result, checked.

    ``romanize_string`` returns a string in its default format but is typed to
    also return a list of edges (another output format). We never ask for
    that format, so anything but a string means uroman changed underneath us:
    fail loudly rather than feed a list to a tokenizer.
    """
    if not isinstance(result, str):
        raise TypeError(f"uroman returned {type(result).__name__}, expected str")
    return result


def is_ascii(text: str) -> bool:
    """True when romanization would be a no-op because the text is already ASCII."""
    # ord() is a character's code point; < 128 is the ASCII range. all() short
    # circuits on the first non-ASCII character, so this is cheap on the common
    # path. Lets callers skip romanizing English rather than paying for it.
    return all(ord(ch) < 128 for ch in text)


def romanize(text: str, lcode: Optional[str] = None) -> Optional[str]:
    """
    Romanize ``text``, or return ``None`` when no romanizer is installed.

    ``lcode`` is an ISO-639-3 hint. uroman works without it, but transliterates
    several scripts more accurately when it is supplied.

    Returning ``None`` rather than the input unchanged is deliberate: the
    caller must be able to tell "romanized" from "could not romanize", because
    feeding un-romanized Devanagari to the aligner is exactly the silent
    degradation golden rule 1 is about.
    """
    romanizer = get_romanizer()
    if romanizer is None:
        return None
    if lcode:
        try:
            result = romanizer.romanize_string(text, lcode=lcode)
        except TypeError:
            # Older uroman builds take no lcode keyword.
            #
            # Catching TypeError to detect an unsupported signature is a
            # version-compatibility shim. It is narrow enough to be safe here
            # because only the call itself sits inside the try: as_text below
            # raises TypeError too, and inside the try that would be mistaken
            # for "old uroman" and silently retried without the hint.
            result = romanizer.romanize_string(text)
        return as_text(result)
    return as_text(romanizer.romanize_string(text))


def reset_cache() -> None:
    """Drop the cached instance. For tests that patch the import."""
    # Without this, the first test to touch the romanizer would fix the cached
    # value for the whole process and later tests could not patch the import.
    # A seam for testability, not production code.
    global _romanizer, _checked
    _romanizer = None
    _checked = False
