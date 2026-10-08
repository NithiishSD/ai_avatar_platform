"""
Language registry for Phase 3 multilingual synthesis.

Backs the MMS-TTS integration with the official Meta MMS TTS language table
(1077 ISO-639-3 codes) plus an ISO-639-1/639-2b alias map so callers can keep
sending familiar BCP-47 codes like ``en``, ``hi-IN`` or ``pt-BR``.

The table is data, not code: ``backend/data/mms_tts_languages.json`` is
regenerated from
``https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html`` and the SIL
ISO-639-3 code table. Nothing here downloads anything at runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parent / "data" / "mms_tts_languages.json"

# Kokoro is English-only; these route to the fast path.
ENGLISH_CODES = {"en", "en-us", "en-gb", "en-au", "en-ca", "en-in", "eng"}

# Languages XTTS-v2 can clone into, keyed by ISO-639-1.
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi",
}

# XTTS-v2's own spelling of a language, where it differs from ISO-639-1:
# Coqui lists Chinese as "zh-cn".
_XTTS_SPELLING = {"zh": "zh-cn"}

# Scripts MMS-TTS cannot tokenize directly; the checkpoint's tokenizer sets
# ``is_uroman`` and the text must be romanized before synthesis.
_NON_LATIN_HINT = re.compile(
    r"[Ͱ-᳿Ḁ-ỿⰀ-퟿豈-﫿]"
)


@dataclass(frozen=True)
class LanguageInfo:
    """Resolved description of one requested language code."""

    requested: str
    iso3: str
    name: str
    mms_supported: bool
    mms_model: Optional[str]
    xtts_supported: bool
    is_english: bool

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "iso3": self.iso3,
            "name": self.name,
            "mmsSupported": self.mms_supported,
            "mmsModel": self.mms_model,
            "xttsSupported": self.xtts_supported,
            "isEnglish": self.is_english,
        }


@lru_cache(maxsize=1)
def _load() -> dict:
    with DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mms_languages() -> Dict[str, str]:
    return _load()["languages"]


def _aliases() -> Dict[str, str]:
    return _load()["aliases"]


def _iso_names() -> Dict[str, str]:
    return _load()["isoNames"]


def normalize_code(code: str) -> str:
    """Lowercase a code and drop region/script subtags: ``pt-BR`` -> ``pt``."""
    if not code:
        return ""
    return re.split(r"[-_]", code.strip().lower())[0]


def to_iso3(code: str) -> Optional[str]:
    """Resolve any ISO-639-1/639-2b/639-3 code to ISO-639-3, or None."""
    base = normalize_code(code)
    if not base:
        return None
    if len(base) == 3:
        if base in _mms_languages() or base in _iso_names():
            return base
        return _aliases().get(base)
    return _aliases().get(base)


def xtts_code(code: str) -> Optional[str]:
    """
    The code XTTS-v2 expects for a language, or None if it cannot speak it.

    The studio sends ISO-639-3 codes (``spa``, ``hin``) because that is how the
    MMS-TTS catalogue is keyed, while XTTS-v2 takes ISO-639-1 (``es``, ``hi``).
    Passing ``spa`` through used to reach the model unchanged, and
    ``xtts_supported`` called Spanish unsupported. Both spellings now resolve.
    """
    iso3 = to_iso3(code)
    if not iso3:
        return None
    for candidate in sorted(XTTS_LANGUAGES):
        if to_iso3(candidate) == iso3:
            return _XTTS_SPELLING.get(candidate, candidate)
    return None


def is_english(code: str) -> bool:
    return (code or "").strip().lower() in ENGLISH_CODES or to_iso3(code) == "eng"


def resolve(code: str) -> LanguageInfo:
    """
    Describe a requested language code.

    Unknown codes still return a LanguageInfo (with ``mms_supported=False``) so
    the router can fall back instead of raising on a typo.
    """
    requested = (code or "").strip()
    iso3 = to_iso3(requested)
    mms = _mms_languages()
    supported = bool(iso3) and iso3 in mms
    if iso3:
        name = mms.get(iso3) or _iso_names().get(iso3) or iso3
    else:
        name = requested or "unknown"
    return LanguageInfo(
        requested=requested,
        iso3=iso3 or "",
        name=name,
        mms_supported=supported,
        mms_model=f"facebook/mms-tts-{iso3}" if supported else None,
        xtts_supported=xtts_code(requested) is not None,
        is_english=is_english(requested),
    )


def mms_model_id(code: str) -> Optional[str]:
    """HuggingFace repo id for a language's MMS-TTS checkpoint, if one exists."""
    return resolve(code).mms_model


def needs_romanization(code: str, text: str = "") -> bool:
    """
    True when the MMS checkpoint for this language is likely a uroman model.

    The authoritative answer lives on the loaded tokenizer (``is_uroman``);
    this is the cheap pre-check used for UI hints and routing warnings.
    """
    if _NON_LATIN_HINT.search(text or ""):
        return True
    return False


def search(query: str = "", limit: int = 50) -> List[LanguageInfo]:
    """Substring search over MMS language names and codes, name-sorted."""
    needle = (query or "").strip().lower()
    matches = [
        (name, iso3)
        for iso3, name in _mms_languages().items()
        if not needle or needle in name.lower() or needle in iso3
    ]
    matches.sort()
    if limit > 0:
        matches = matches[:limit]
    return [resolve(iso3) for _, iso3 in matches]


def supported_count() -> int:
    return len(_mms_languages())


def catalogue_source() -> str:
    return _load()["source"]
