"""
The protected-voice list: voices that must not be cloned (R-46, T8.9).

A person who does not want their voice cloned (or who withdraws consent) can have a recording of it
registered here. Only its **speaker embedding** (192 numbers from ECAPA-TDNN) is kept, under an opaque
id; the audio is discarded and no name is stored. Two checks use the list:

* **before** a clone, the reference recording is compared with every protected voice; a match is
  refused, written to the audit trail as ``abuse_alert`` and ``voice_refused``, and never synthesised;
* **after** a clone, the generated clip is compared too, because a consented reference can still
  produce a voice that sounds like a protected person; that is raised as an ``abuse_alert`` (the clip
  exists, so the alert is the record, and it names the protected id and the similarity).

With an empty list nothing is computed, so the default path pays nothing. The threshold is a cosine
similarity of ECAPA embeddings (``PROTECTED_VOICE_THRESHOLD``, default in ``DEFAULT_THRESHOLD``); how
it was chosen from measured same-speaker / different-speaker / clone similarities is in
``docs/11-DECISIONS.md``.

What this cannot do: it only knows the voices someone registered; it does not detect a clone of
anyone else, and a determined user can change a voice enough to slip under a threshold.

Concepts, explained once:

* A **speaker embedding** is a fixed-length vector that a speaker-recognition
  network (here ECAPA-TDNN) produces from a recording. It describes *how the
  voice sounds*, not what was said, so two clips of the same person land close
  together and different people land further apart. It cannot be turned back
  into audio.
* **Cosine similarity** compares two vectors by the angle between them: 1.0
  means the same direction, 0 means unrelated. Loudness and clip length change
  a vector's length, not its direction, which is why the angle is used.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import audit_log
from job_store import JobStore

# The ``kind`` these records use in the shared SQLite JobStore.
KIND = "protected_voice"
# Measured (D-51 in docs/11-DECISIONS.md): above the highest different-voice
# pair seen (0.308) and below the weakest English clone (0.55).
DEFAULT_THRESHOLD = 0.4


def threshold() -> float:
    """The match threshold: ``$PROTECTED_VOICE_THRESHOLD`` or ``DEFAULT_THRESHOLD``."""
    return float(os.getenv("PROTECTED_VOICE_THRESHOLD", DEFAULT_THRESHOLD))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, in [-1, 1]."""
    # The tiny epsilon keeps an all-zero vector from dividing by zero.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class ProtectedVoices:
    """The registered voices, stored as embeddings in SQLite, and the checks against them."""
    def __init__(self, store: Optional[JobStore] = None, embed: Any = None) -> None:
        self._store = store or JobStore()
        # ``embed(path) -> np.ndarray``; injected so tests do not load ECAPA, and shared with the auditor in the app.
        self._embed = embed
        self._lock = threading.Lock()

    def _embedder(self):
        """The embedding function, loading ECAPA through the quality auditor on first use."""
        if self._embed is None:
            from quality_auditor import SpeechQualityAuditor

            self._embed = SpeechQualityAuditor().embed_speaker
        return self._embed

    def list(self) -> List[Dict[str, Any]]:
        """Active protected voices as ``{id, added}``; embeddings are never returned."""
        return [{"id": voice_id, "added": rec["added"]} for voice_id, rec in self._store.all(KIND) if not rec.get("removed")]

    def _embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """``(id, embedding)`` for every active protected voice; removed ones are skipped."""
        return [(voice_id, np.asarray(rec["embedding"], dtype=np.float32))
                for voice_id, rec in self._store.all(KIND) if not rec.get("removed")]

    def register(self, audio_path: Path | str) -> str:
        """Keep the speaker embedding of this recording and return its opaque id. The audio is not kept."""
        embedding = self._embedder()(audio_path)
        # Random and short: the id must not reveal who the voice belongs to.
        voice_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._store.put(KIND, voice_id, {"embedding": [float(x) for x in embedding], "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        audit_log.shared_audit().record("protected_voice_added", subject=voice_id, basis=None)
        return voice_id

    def remove(self, voice_id: str) -> bool:
        """Withdraw a protected voice. False if it was unknown or already removed."""
        record = self._store.get(KIND, voice_id)
        if record is None or record.get("removed"):
            return False
        # Kept as a tombstone with the embedding blanked: ids stay unique, and the biometric data is gone.
        self._store.put(KIND, voice_id, {"embedding": [], "added": record["added"], "removed": True})
        audit_log.shared_audit().record("protected_voice_removed", subject=voice_id, basis=None)
        return True

    def best_match(self, audio_path: Path | str) -> Optional[Tuple[str, float]]:
        """The closest protected voice to this recording as ``(id, similarity)``; ``None`` if the list is empty."""
        protected = self._embeddings()
        if not protected:
            return None
        embedding = self._embedder()(audio_path)
        return max(((voice_id, cosine(embedding, other)) for voice_id, other in protected), key=lambda pair: pair[1])

    def check(self, audio_path: Path | str, *, kind: str, subject: str) -> Optional[Dict[str, Any]]:
        """
        Compare a recording with the list. If it is at or above the threshold, write an ``abuse_alert``
        (``kind`` says whether this was the reference or the output) and return the alert details.
        """
        found = self.best_match(audio_path)
        if found is None or found[1] < threshold():
            return None
        voice_id, similarity = found
        details = {"kind": kind, "protectedId": voice_id, "similarity": round(similarity, 3), "threshold": threshold()}
        audit_log.shared_audit().record("abuse_alert", subject=subject, basis=None, **details)
        return details


_shared: Optional[ProtectedVoices] = None


def shared() -> ProtectedVoices:
    """The process-wide list, created on first use."""
    global _shared
    if _shared is None:
        _shared = ProtectedVoices()
    return _shared
