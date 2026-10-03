"""
Avatar store -- the registered faces the renderer is allowed to animate.

``AvatarRenderJob.avatarId`` was an opaque string until now. This module gives
it a meaning: an avatar is an image in ``inputs/faces/`` named ``<avatarId>.png``
with a provenance sidecar beside it. Nothing else is an avatar, and that is
deliberate -- the store is the single door a face comes through, so it is the
single place the two rules about faces are enforced:

* **Consent before use.** ``register`` refuses a human face with no consent
  basis, and ``require_usable`` refuses to hand the renderer an image whose
  sidecar is missing or does not permit use. A file copied into the folder by
  hand is therefore listed, but cannot be rendered until it is registered.
* **The quality gate.** A photo with no face, two faces, or a face turned too
  far is rejected at registration with a message the uploader can act on,
  instead of producing a broken video later.

Every stored image is re-encoded to PNG. That strips EXIF (which can carry
location and device data the uploader did not mean to share) after first
reading any embedded copyright notice out of it, so a notice like the one that
forced the first test photo to be deleted is surfaced instead of lost.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import provenance
from face_engine import (
    FACE_ENGINE_LOCK,
    FaceMeshEngine,
    FaceQualityReport,
    shared_face_engine,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACES_DIR = PROJECT_ROOT / "inputs" / "faces"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
AVATAR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Larger uploads are scaled down: the renderer caps output at 1080p, and a
# 40-megapixel phone photo would only cost memory.
MAX_IMAGE_SIDE = 2048
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Checked from the header, before any pixel is decoded: a few kilobytes of
# PNG can declare a gigapixel canvas.
MAX_IMAGE_PIXELS = 40_000_000

# Text chunks (PNG tEXt, JPEG comment, XMP) that can carry a rights notice.
_INFO_NOTICE_KEYS = ("copyright", "author", "artist", "description", "comment", "rights")
_EXIF_SUB_IFD = 0x8769
_EXIF_USER_COMMENT = 0x9286

# EXIF tags that can carry rights or usage text a human should read.
_EXIF_NOTICE_TAGS = {
    0x8298: "copyright",
    0x010E: "imageDescription",
    0x013B: "artist",
    0x9286: "userComment",
}


class AvatarError(ValueError):
    """A registration request that cannot be honoured; the message says why."""


class AvatarNotFound(LookupError):
    """No avatar with that id exists in the store."""


class AvatarConsentError(PermissionError):
    """The image exists but its provenance does not permit using it."""


class AvatarRejected(AvatarError):
    """The photo failed the quality gate. ``report`` holds the reasons."""

    def __init__(self, report: FaceQualityReport):
        self.report = report
        super().__init__(
            "; ".join(issue.message for issue in report.errors)
            or "photo rejected by the quality gate"
        )


@dataclass
class AvatarRecord:
    """One registered (or merely present) face image."""

    avatar_id: str
    path: Path
    width: int
    height: int
    usable: bool
    usability_reason: str
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "avatarId": self.avatar_id,
            "filename": self.path.name,
            "imageUrl": f"/api/v1/avatar/faces/{self.avatar_id}/image",
            "width": self.width,
            "height": self.height,
            "usable": self.usable,
            "usabilityReason": self.usability_reason,
            "provenance": self.provenance,
        }


def validate_avatar_id(avatar_id: str) -> str:
    """
    Return ``avatar_id`` if it is safe to use as a filename.

    The id becomes a path component, so anything outside a conservative
    character set is refused rather than sanitised: silently rewriting
    ``../x`` to ``x`` would register an avatar under a name nobody asked for.
    """
    if not isinstance(avatar_id, str) or not AVATAR_ID_PATTERN.match(avatar_id):
        raise AvatarError(
            "avatarId must be 1-64 characters of letters, digits, '_' or '-', "
            f"starting with a letter or digit (got {avatar_id!r})"
        )
    return avatar_id


def _embedded_notices(image) -> Dict[str, str]:
    """Rights-related text found in the image's EXIF, if any."""
    notices: Dict[str, str] = {}
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - a malformed EXIF block is not fatal
        return notices

    def keep(label: str, value) -> None:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str) and value.strip("\x00 \t\r\n"):
            notices.setdefault(label, value.strip("\x00 \t\r\n")[:500])

    for tag, label in _EXIF_NOTICE_TAGS.items():
        keep(label, exif.get(tag))
    # UserComment lives in the Exif sub-IFD, behind an 8-byte charset marker.
    try:
        comment = exif.get_ifd(_EXIF_SUB_IFD).get(_EXIF_USER_COMMENT)
    except Exception:  # noqa: BLE001
        comment = None
    if isinstance(comment, bytes) and len(comment) > 8:
        comment = comment[8:]
    keep("userComment", comment)
    # Rights text outside EXIF: PNG text chunks, JPEG comments, XMP.
    for key, value in getattr(image, "info", {}).items():
        lowered = str(key).lower()
        if any(name in lowered for name in _INFO_NOTICE_KEYS):
            keep(str(key), value)
        elif lowered in ("xmp", "xml:com.adobe.xmp") and value:
            text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
            match = re.search(r"<dc:rights>(.*?)</dc:rights>", text, re.S)
            if match:
                keep("xmpRights", re.sub(r"<[^>]+>", " ", match.group(1)))
    return notices


def decode_image(image: Union[bytes, bytearray, str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Decode to an upright RGB uint8 array, capped at ``MAX_IMAGE_SIDE``.

    Returns the array and any rights notices embedded in the file.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        elif array.ndim != 3 or array.shape[2] != 3:
            raise AvatarError(f"unsupported image array shape {image.shape}")
        handle = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        notices: Dict[str, str] = {}
    else:
        try:
            if isinstance(image, (bytes, bytearray)):
                if len(image) > MAX_UPLOAD_BYTES:
                    raise AvatarError(
                        f"image is {len(image) / 1e6:.1f} MB; the limit is "
                        f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB"
                    )
                handle = Image.open(io.BytesIO(bytes(image)))
            else:
                path = Path(image)
                if not path.is_file():
                    raise AvatarError(f"image not found: {path}")
                handle = Image.open(path)
            if handle.width * handle.height > MAX_IMAGE_PIXELS:
                raise AvatarError(
                    f"image is {handle.width}x{handle.height} pixels; the limit is "
                    f"{MAX_IMAGE_PIXELS // 1_000_000} megapixels"
                )
            handle.load()
        except UnidentifiedImageError as err:
            raise AvatarError(
                "the file is not an image this project can read "
                f"(supported: {', '.join(IMAGE_EXTENSIONS)})"
            ) from err
        except (OSError, Image.DecompressionBombError) as err:
            # Truncated or corrupt data, or a canvas PIL itself refuses.
            raise AvatarError(f"the image could not be decoded: {err}") from err
        notices = _embedded_notices(handle)
        handle = ImageOps.exif_transpose(handle).convert("RGB")

    longest = max(handle.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        handle = handle.resize(
            (max(1, round(handle.width * scale)), max(1, round(handle.height * scale))),
            Image.LANCZOS,
        )
    return np.ascontiguousarray(np.asarray(handle, dtype=np.uint8)), notices


class AvatarStore:
    """Filesystem-backed registry of avatar images and their provenance."""

    def __init__(
        self,
        root: Optional[Path | str] = None,
        engine: Optional[FaceMeshEngine] = None,
    ) -> None:
        self.root = Path(root) if root is not None else FACES_DIR
        self._engine = engine

    # -- helpers ------------------------------------------------------------

    def _engine_or_shared(self) -> FaceMeshEngine:
        return self._engine if self._engine is not None else shared_face_engine()

    def _find(self, avatar_id: str) -> Optional[Path]:
        # Same rule as ``list``: the extension is matched case-insensitively,
        # so whatever is listed can also be fetched and deleted.
        if not self.root.is_dir():
            return None
        for path in sorted(self.root.iterdir()):
            if path.stem == avatar_id and path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                return path
        return None

    def _record(self, avatar_id: str, path: Path) -> AvatarRecord:
        width = height = 0
        try:
            from PIL import Image

            with Image.open(path) as handle:
                width, height = handle.size
        except Exception as err:  # noqa: BLE001
            logger.warning("Could not read avatar image %s: %s", path, err)
        usable, reason = provenance.usability(path)
        return AvatarRecord(
            avatar_id=avatar_id,
            path=path,
            width=width,
            height=height,
            usable=usable,
            usability_reason=reason,
            provenance=provenance.describe(path),
        )

    # -- reads --------------------------------------------------------------

    def list(self) -> List[AvatarRecord]:
        """Every image in the store, registered or not, sorted by id."""
        if not self.root.is_dir():
            return []
        records: List[AvatarRecord] = []
        seen = set()
        for path in sorted(self.root.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            avatar_id = path.stem
            if avatar_id in seen or not AVATAR_ID_PATTERN.match(avatar_id):
                continue
            seen.add(avatar_id)
            records.append(self._record(avatar_id, path))
        return records

    def get(self, avatar_id: str) -> AvatarRecord:
        validate_avatar_id(avatar_id)
        path = self._find(avatar_id)
        if path is None:
            raise AvatarNotFound(
                f"avatar {avatar_id!r} is not registered. Register a face with "
                "scripts/make_avatar.py (or POST /api/v1/avatar/faces), then use "
                "its avatarId."
            )
        return self._record(avatar_id, path)

    def require_usable(self, avatar_id: str) -> AvatarRecord:
        """``get``, but refuse an image whose provenance does not permit use."""
        record = self.get(avatar_id)
        if not record.usable:
            raise AvatarConsentError(
                f"avatar {avatar_id!r} cannot be used: {record.usability_reason}"
            )
        return record

    def load_image(self, avatar_id: str) -> np.ndarray:
        """The avatar as an RGB array, after the consent check."""
        record = self.require_usable(avatar_id)
        from PIL import Image

        with Image.open(record.path) as handle:
            return np.ascontiguousarray(handle.convert("RGB"))

    # -- writes -------------------------------------------------------------

    def register(
        self,
        image: Union[bytes, bytearray, str, Path, np.ndarray],
        avatar_id: str,
        source: str,
        subject: str = "",
        licence: str = "",
        consent_basis: str = "",
        notes: str = "",
        extra: Optional[Dict[str, object]] = None,
        overwrite: bool = False,
        check_quality: bool = True,
    ) -> Tuple[AvatarRecord, Optional[FaceQualityReport]]:
        """
        Validate, store and record an avatar image.

        Raises ``AvatarError`` for a bad id, a missing consent basis or an
        unreadable file, and ``AvatarRejected`` when the quality gate fails.
        Nothing is written unless every check passes.
        """
        validate_avatar_id(avatar_id)
        if source not in (provenance.HUMAN, provenance.SYNTHETIC):
            raise AvatarError(
                f"source must be {provenance.HUMAN!r} or {provenance.SYNTHETIC!r}"
            )
        if source == provenance.HUMAN:
            if consent_basis not in provenance.FACE_CONSENT_BASES:
                raise AvatarError(
                    "a photo of a real person needs a consent basis, one of "
                    f"{sorted(provenance.FACE_CONSENT_BASES)}. Do not register a "
                    "face whose owner has not agreed to it being animated."
                )
            if not subject.strip():
                raise AvatarError(
                    "a photo of a real person needs the subject's name (who is "
                    "in the picture), so the consent record means something"
                )

        existing = self._find(avatar_id)
        if existing is not None and not overwrite:
            raise AvatarError(
                f"avatar {avatar_id!r} already exists; choose another avatarId"
            )

        array, notices = decode_image(image)

        report: Optional[FaceQualityReport] = None
        if check_quality:
            with FACE_ENGINE_LOCK:
                report = self._engine_or_shared().check_quality(array)
            if not report.passed:
                raise AvatarRejected(report)

        from PIL import Image

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{avatar_id}.png"
        # Fail closed: the old record goes before the new pixels arrive. If
        # anything below fails, the image has no sidecar and cannot be used,
        # instead of a new face inheriting the previous subject's consent.
        if existing is not None:
            provenance.sidecar_path(existing).unlink(missing_ok=True)
            if existing != path:
                existing.unlink()
        provenance.sidecar_path(path).unlink(missing_ok=True)
        Image.fromarray(array, "RGB").save(path, format="PNG")

        lineage = dict(extra or {})
        if notices:
            # Kept in the record and logged: a rights notice embedded in the
            # file outranks whatever the uploader typed into the form.
            lineage["embeddedNotices"] = notices
            logger.warning(
                "Avatar %s carries embedded rights text; a human must check it "
                "permits this use: %s",
                avatar_id,
                notices,
            )
        provenance.write(
            path,
            source=source,
            speaker=subject.strip() or ("synthetic" if source == provenance.SYNTHETIC else "unknown"),
            licence=licence,
            consent_basis=consent_basis,
            notes=notes,
            extra=lineage,
        )
        logger.info("Registered avatar %s (%s) at %s", avatar_id, source, path)
        return self._record(avatar_id, path), report

    def delete(self, avatar_id: str) -> None:
        """Remove an avatar and its provenance record."""
        record = self.get(avatar_id)
        record.path.unlink(missing_ok=True)
        provenance.sidecar_path(record.path).unlink(missing_ok=True)
        logger.info("Deleted avatar %s", avatar_id)
