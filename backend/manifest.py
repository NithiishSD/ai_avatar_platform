"""
Signed provenance manifests (R-33, R-36).

A watermark says "this came from a system holding our key". It cannot say what
went into the video or on what basis. The manifest does: it lists the inputs
(the avatar face, the voice reference, the speech model), the models and
settings used, and the consent basis of every recording and photo, and it is
**signed** and **bound to the exact file** by the file's SHA-256.

    manifest = { ...what went in..., "content": {"videoSha256": ...}, "signature": {...} }

Signing is Ed25519 (the ``cryptography`` package). The private key is derived from
the platform secret (``watermark_engine.signing_key``), so one secret governs both the
watermark and the signature; the *public* key travels in every manifest and can be
fetched from the API, so anyone can check a signature without being able to make one.
A signature proves the manifest was issued by whoever holds the key and has not been
edited; whether that key is *ours* is decided by comparing the public key with the
verifier's own.

Names are withheld: the manifest carries the consent basis, the source and a hash of
each reference file, but not the name of the person in a photo or on a recording. The
sidecar next to the original file holds that, under the same consent rules.

What this is not: it is not C2PA. C2PA is the industry's container format with a
certificate chain; this is a simpler signed JSON with the same purpose. Importing one
into a C2PA store is a future step, recorded in ``docs/11-DECISIONS.md``.

Where it sits in the pipeline:
  1. Speech synthesis (``voice_engine.py``, ``voice_to_avatar.py``) writes a *speech record* next to
     each audio clip with :func:`write_speech_record`.
  2. When a render finishes, ``render_engine.py`` draws a :func:`new_manifest_id`, embeds it in the
     video watermark, and calls :func:`build_manifest`, which reads the speech record back.
  3. ``authenticity.py`` calls :func:`verify` when someone asks whether a file is genuine.

Concepts used here:
  * *Ed25519* is a public-key signature scheme. The private key makes a 64-byte signature over some
    bytes; the matching public key checks it. Changing one byte of the signed content, or using any
    other key, makes the check fail. Keys are 32 bytes, so they fit in a JSON field as hex.
  * *SHA-256 binding*. The manifest stores the video file's SHA-256 inside the signed content. The
    signature protects that hash, and the hash pins the exact bytes of the file, so the manifest
    describes one file only. A re-encoded copy has a different hash.
  * *Canonical form*. JSON can spell the same object many ways (key order, spaces). A signature is
    over bytes, so signer and verifier must produce identical bytes; :func:`canonical` fixes one form.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import provenance
import watermark_engine

logger = logging.getLogger(__name__)

# A versioned schema name: a future layout gets "/2", and ``verify`` rejects anything it does not know.
SCHEMA = "ai-avatar-platform/manifest/1"
# Written into each signature so a verifier knows how to check it, and can refuse an unknown one.
ALGORITHM = "ed25519"
# The speech record sits beside its clip: "speech.wav" -> "speech.wav.speech.json".
SPEECH_RECORD_SUFFIX = ".speech.json"
PLATFORM = "ai-avatar-platform"


# --------------------------------------------------------------------------- keys and canonical form
def _private_key():
    """The Ed25519 signing key, derived on each call from the platform secret.

    Nothing is stored: the same secret always gives the same key, so manifests stay verifiable
    across restarts as long as the secret is kept.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    # HMAC-SHA256 with a fixed label turns the secret into 32 bytes, the size of an Ed25519 seed.
    # The label separates this use from the watermark's use of the same secret, so knowing one
    # derived value tells nothing about the other.
    seed = hmac.new(watermark_engine.signing_key(), b"avatar-platform/manifest-signing/v1", hashlib.sha256).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_key_hex() -> str:
    """The platform's public key as 64 hex characters: safe to publish, it can only verify."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return _private_key().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def canonical(obj: Dict[str, Any]) -> bytes:
    """The bytes that are signed: sorted keys, no whitespace, UTF-8. Any edit changes them."""
    # separators=(",", ":") drops the default spaces; ensure_ascii=False keeps non-ASCII as UTF-8
    # instead of \u escapes, so the bytes do not depend on that setting either.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file's bytes, read in 4 MiB chunks so a large video never sits in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        # iter(callable, sentinel) calls read() until it returns b"" (end of file). 1 << 22 = 4 MiB.
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_manifest_id() -> bytes:
    """16 random bytes: also written into the video's watermark, so a copy can be traced back.

    ``secrets`` rather than ``random``: it draws from the OS's cryptographic source, so ids cannot
    be predicted. 128 bits also makes two ids colliding by chance negligible.
    """
    return secrets.token_bytes(16)


# --------------------------------------------------------------------------- the speech record
def speech_record_path(audio_path: Path | str) -> Path:
    """Where the speech record for ``audio_path`` lives (the audio path plus a suffix)."""
    return Path(str(audio_path) + SPEECH_RECORD_SUFFIX)


def reference_summary(speaker_wav: Optional[str]) -> Optional[Dict[str, Any]]:
    """What may be said about a voice reference without naming the speaker."""
    if not speaker_wav or not Path(speaker_wav).is_file():
        # No reference: the clip used a stock voice, so there is nothing to describe.
        return None
    # The provenance sidecar beside the recording holds its source and consent basis (golden rule 3).
    described = provenance.describe(speaker_wav)
    return {
        "sha256": sha256_file(speaker_wav),
        "source": described.get("source"),
        "consentBasis": described.get("consentBasis") or None,
        "licence": described.get("licence") or None,
        "admissibleAsEvidence": bool(described.get("admissible")),
    }


def write_speech_record(
    audio_path: Path | str,
    *,
    model: str,
    mode: str,
    language: str,
    speaker_wav: Optional[str],
    clone_engine: Optional[str],
    emotion: Optional[Dict[str, Any]],
    alignment_method: Optional[str],
    duration_seconds: float,
    watermark: Optional[Dict[str, Any]],
    origin: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Record how a clip was made, next to the clip, tied to its exact bytes.

    The render step later reads this to fill the video's manifest. If the clip is replaced
    (``speech.wav`` is overwritten by the next synthesis) the hash stops matching and the
    manifest says the record could not be used, instead of describing the wrong audio.
    """
    record = {
        "schema": "ai-avatar-platform/speech-record/1",
        # "generated" by this platform's speech engines, or a dict describing audio a caller supplied
        # (voice-to-avatar): the manifest must not present someone's own recording as synthetic speech.
        "origin": origin or {"type": "generated"},
        "audioSha256": sha256_file(audio_path),
        "model": model,
        "mode": mode,
        "language": language,
        "cloneEngine": clone_engine,
        "voiceReference": reference_summary(speaker_wav),
        # Only the dominant emotion and its strength; the full emotion vector is not needed for provenance.
        "emotion": {"dominant": (emotion or {}).get("dominant"), "intensity": (emotion or {}).get("intensity")} if emotion else None,
        "alignmentMethod": alignment_method,
        "durationSeconds": round(float(duration_seconds), 3),
        "audioWatermark": watermark,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = speech_record_path(audio_path)
    # Indented for a person to read; this file is not signed, so its byte form does not matter.
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path


def read_speech_record(audio_path: Path | str) -> Dict[str, Any]:
    """The record for this exact audio file, or ``{"status": ...}`` saying why there is none."""
    path = speech_record_path(audio_path)
    if not path.is_file():
        return {"status": "unrecorded", "reason": "no speech record exists for this audio (it was not made by this server's synthesis, or predates records)"}
    try:
        record = json.loads(path.read_text())
    except ValueError:
        return {"status": "unreadable", "reason": "the speech record is not valid JSON"}
    # The same SHA-256 binding as the manifest: the record only describes the bytes it was written for.
    if record.get("audioSha256") != sha256_file(audio_path):
        return {"status": "mismatched", "reason": "the audio changed after its speech record was written (the file was overwritten or edited)"}
    return {"status": "matched", **record}


# --------------------------------------------------------------------------- build, sign, verify
def build_manifest(
    *,
    manifest_id: bytes,
    video_path: Path | str,
    audio_path: Path | str,
    avatar: Dict[str, Any],
    render: Dict[str, Any],
    video_watermark: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble and sign the manifest for a finished video. ``avatar`` is an ``AvatarRecord.to_dict()``.

    ``render`` is the render result (size, fps, engine, label); ``video_watermark`` is what was
    embedded in the frames; ``models`` names every model used. Returns the signed manifest dict.
    """
    avatar_provenance = avatar.get("provenance") or {}
    unsigned = {
        "schema": SCHEMA,
        "manifestId": manifest_id.hex(),
        "issuedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The public key travels with the manifest so it can be checked offline, without the API.
        "issuer": {"platform": PLATFORM, "publicKey": public_key_hex(), "algorithm": ALGORITHM},
        # Always True: every video this platform renders is synthetic, and the manifest says so plainly.
        "aiGenerated": True,
        "content": {
            "videoSha256": sha256_file(video_path),
            "videoBytes": Path(video_path).stat().st_size,
            **{k: render.get(k) for k in ("width", "height", "fps", "frameCount", "durationSeconds")},
        },
        "inputs": {
            "avatar": {
                "avatarId": avatar.get("avatarId"),
                "source": avatar_provenance.get("source"),
                "consentBasis": avatar_provenance.get("consentBasis") or None,
                "licence": avatar_provenance.get("licence") or None,
                "imageSha256": _avatar_hash(avatar),
                "generator": (avatar_provenance.get("extra") or {}).get("generator"),
                "seed": (avatar_provenance.get("extra") or {}).get("seed"),
            },
            "audio": {"sha256": sha256_file(audio_path), "speechRecord": read_speech_record(audio_path)},
        },
        "processing": {
            "renderEngine": render.get("engine"),
            "quality": render.get("quality"),
            "background": render.get("background"),
            "visibleLabel": render.get("label"),
        },
        "models": models,
        "watermarks": {"video": video_watermark},
    }
    # Everything above is inside the signature; only the "signature" key itself is added after.
    return sign(unsigned)


def _avatar_hash(avatar: Dict[str, Any]) -> Optional[str]:
    """SHA-256 of the avatar's stored image, or ``None`` when it cannot be found."""
    try:
        # avatarId -> the stored image; the store resolves it, this just hashes what it returns
        from avatar_store import AvatarStore

        return sha256_file(AvatarStore().get(str(avatar.get("avatarId"))).path)
    except Exception:  # noqa: BLE001 - a missing image must not stop a manifest being issued; it says so
        return None


def sign(unsigned: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``unsigned`` plus a ``signature`` entry over its canonical bytes (base64-encoded)."""
    signature = _private_key().sign(canonical(unsigned))
    return {**unsigned, "signature": {"algorithm": ALGORITHM, "value": base64.b64encode(signature).decode("ascii")}}


@dataclass
class ManifestReport:
    """The outcome of :func:`verify`: three separate checks, plus a plain-language problem list.

    They are kept apart because each answers a different question; a valid signature alone only
    says "unedited", not "ours" and not "this file".
    """

    valid_signature: bool                 # the manifest is unedited and signed by the key it names
    issued_by_us: bool                    # that key is this platform's key
    matches_file: Optional[bool]          # the video's SHA-256 equals the manifest's (None: no file given)
    manifest_id: Optional[str]
    problems: List[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """All checks pass. ``matches_file`` of ``None`` (no file given) does not count against it."""
        return self.valid_signature and self.issued_by_us and self.matches_file is not False

    def to_dict(self) -> Dict[str, Any]:
        """The report as the API's camelCase JSON."""
        return {
            "validSignature": self.valid_signature,
            "issuedByUs": self.issued_by_us,
            "matchesFile": self.matches_file,
            "manifestId": self.manifest_id,
            "trustworthy": self.trustworthy,
            "problems": self.problems,
        }


def verify(manifest: Dict[str, Any], video_path: Path | str | None = None) -> ManifestReport:
    """
    Check a manifest: schema, signature, that the signer is us, and (given the file) that it is this file.

    Never raises on a bad manifest: every failure is a ``problems`` entry, so a caller can show them.
    """
    problems: List[str] = []
    # The input is untrusted JSON from a caller, so its type is checked before any .get().
    manifest_id = manifest.get("manifestId") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        return ManifestReport(False, False, None, manifest_id, ["not a manifest of this platform (unknown or missing schema)"])
    signature = manifest.get("signature") or {}
    # Rebuild exactly what was signed: the manifest minus its signature entry.
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    issuer_key = ((manifest.get("issuer") or {}).get("publicKey")) or ""
    valid = False
    if signature.get("algorithm") != ALGORITHM:
        problems.append(f"unsupported signature algorithm {signature.get('algorithm')!r}")
    else:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        # Verification uses the key the manifest *names*; whether that key is ours is checked below.
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(issuer_key)).verify(
                base64.b64decode(signature.get("value", "")), canonical(unsigned)
            )
            valid = True
        # ValueError/TypeError cover malformed hex or base64, which is just another failed check.
        except (InvalidSignature, ValueError, TypeError):
            problems.append("the signature does not match the manifest: it was edited, or signed by a different key than it names")
    # Is the named key this platform's own? compare_digest compares in constant time (see security.py).
    ours = hmac.compare_digest(issuer_key.encode(), public_key_hex().encode())
    if valid and not ours:
        problems.append("signed by a different key than this platform's (a valid manifest, but not issued here)")
    matches: Optional[bool] = None
    if video_path is not None:
        # A plain == is fine here: both hashes are public, so there is no secret to leak through timing.
        matches = sha256_file(video_path) == (manifest.get("content") or {}).get("videoSha256")
        if not matches:
            problems.append("the file's SHA-256 differs from the manifest's: this is not the exact video it was issued for (re-encoded, trimmed or edited)")
    return ManifestReport(valid, ours, matches, manifest_id, problems)
