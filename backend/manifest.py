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

SCHEMA = "ai-avatar-platform/manifest/1"
ALGORITHM = "ed25519"
SPEECH_RECORD_SUFFIX = ".speech.json"
PLATFORM = "ai-avatar-platform"


# --------------------------------------------------------------------------- keys and canonical form
def _private_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    seed = hmac.new(watermark_engine.signing_key(), b"avatar-platform/manifest-signing/v1", hashlib.sha256).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_key_hex() -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return _private_key().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def canonical(obj: Dict[str, Any]) -> bytes:
    """The bytes that are signed: sorted keys, no whitespace, UTF-8. Any edit changes them."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_manifest_id() -> bytes:
    """16 random bytes: also written into the video's watermark, so a copy can be traced back."""
    return secrets.token_bytes(16)


# --------------------------------------------------------------------------- the speech record
def speech_record_path(audio_path: Path | str) -> Path:
    return Path(str(audio_path) + SPEECH_RECORD_SUFFIX)


def reference_summary(speaker_wav: Optional[str]) -> Optional[Dict[str, Any]]:
    """What may be said about a voice reference without naming the speaker."""
    if not speaker_wav or not Path(speaker_wav).is_file():
        return None
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
        "emotion": {"dominant": (emotion or {}).get("dominant"), "intensity": (emotion or {}).get("intensity")} if emotion else None,
        "alignmentMethod": alignment_method,
        "durationSeconds": round(float(duration_seconds), 3),
        "audioWatermark": watermark,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = speech_record_path(audio_path)
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
    """Assemble and sign the manifest for a finished video. ``avatar`` is an ``AvatarRecord.to_dict()``."""
    avatar_provenance = avatar.get("provenance") or {}
    unsigned = {
        "schema": SCHEMA,
        "manifestId": manifest_id.hex(),
        "issuedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issuer": {"platform": PLATFORM, "publicKey": public_key_hex(), "algorithm": ALGORITHM},
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
    return sign(unsigned)


def _avatar_hash(avatar: Dict[str, Any]) -> Optional[str]:
    try:
        # avatarId -> the stored image; the store resolves it, this just hashes what it returns
        from avatar_store import AvatarStore

        return sha256_file(AvatarStore().get(str(avatar.get("avatarId"))).path)
    except Exception:  # noqa: BLE001 - a missing image must not stop a manifest being issued; it says so
        return None


def sign(unsigned: Dict[str, Any]) -> Dict[str, Any]:
    signature = _private_key().sign(canonical(unsigned))
    return {**unsigned, "signature": {"algorithm": ALGORITHM, "value": base64.b64encode(signature).decode("ascii")}}


@dataclass
class ManifestReport:
    valid_signature: bool                 # the manifest is unedited and signed by the key it names
    issued_by_us: bool                    # that key is this platform's key
    matches_file: Optional[bool]          # the video's SHA-256 equals the manifest's (None: no file given)
    manifest_id: Optional[str]
    problems: List[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.valid_signature and self.issued_by_us and self.matches_file is not False

    def to_dict(self) -> Dict[str, Any]:
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
    manifest_id = manifest.get("manifestId") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        return ManifestReport(False, False, None, manifest_id, ["not a manifest of this platform (unknown or missing schema)"])
    signature = manifest.get("signature") or {}
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    issuer_key = ((manifest.get("issuer") or {}).get("publicKey")) or ""
    valid = False
    if signature.get("algorithm") != ALGORITHM:
        problems.append(f"unsupported signature algorithm {signature.get('algorithm')!r}")
    else:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(issuer_key)).verify(
                base64.b64decode(signature.get("value", "")), canonical(unsigned)
            )
            valid = True
        except (InvalidSignature, ValueError, TypeError):
            problems.append("the signature does not match the manifest: it was edited, or signed by a different key than it names")
    ours = hmac.compare_digest(issuer_key.encode(), public_key_hex().encode())
    if valid and not ours:
        problems.append("signed by a different key than this platform's (a valid manifest, but not issued here)")
    matches: Optional[bool] = None
    if video_path is not None:
        matches = sha256_file(video_path) == (manifest.get("content") or {}).get("videoSha256")
        if not matches:
            problems.append("the file's SHA-256 differs from the manifest's: this is not the exact video it was issued for (re-encoded, trimmed or edited)")
    return ManifestReport(valid, ours, matches, manifest_id, problems)
