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

Where it sits in the pipeline: ``app.py`` calls ``register`` when a photo is
uploaded (``POST /api/v1/avatar/faces``) and ``require_usable`` / ``get``
before a job is accepted. ``render_engine``, ``live_engine`` and
``avatar_generator`` then load pixels through ``load_image``, so the consent
check runs on every path that turns a face into video.

Concepts used here, explained once:

**Provenance sidecar.** A small JSON file stored next to the media it
describes, named ``<file>.provenance.json`` (see ``provenance.py``). It
records who is in the picture, where it came from, under what licence and on
what consent basis. Keeping it beside the image (instead of in a database)
means the two travel together: copy or delete one and you see the other.
No sidecar means "not allowed", never "probably fine".

**EXIF.** Metadata a camera or editor writes into a JPEG/PNG: orientation,
GPS position, device model, and sometimes a copyright line. It is invisible
when you look at the photo, which is why we read it out and then drop it.

**Error classes as an API.** Each failure below is its own exception type so
``app.py`` can map it to an HTTP status without parsing messages:
``AvatarRejected`` -> 422, ``AvatarError`` -> 400, ``AvatarNotFound`` -> 404,
``AvatarConsentError`` -> 403.
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

# parents[1] of backend/avatar_store.py is the repository root, so the store
# finds inputs/faces/ no matter which directory the server was started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACES_DIR = PROJECT_ROOT / "inputs" / "faces"

# The formats Pillow decodes here. Matched case-insensitively against the
# file suffix, so "Photo.JPG" counts.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
# A regular expression (regex) is a pattern for text. This one reads:
#   ^ ... $            the whole string must match, not just part of it;
#   [A-Za-z0-9]        the first character is a letter or digit;
#   [A-Za-z0-9_-]{0,63} then up to 63 more of letters, digits, '_' or '-'.
# So 1-64 characters, no '.', '/' or spaces: the id is safe as a filename and
# cannot climb out of the folder with "../".
AVATAR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Larger uploads are scaled down: the renderer caps output at 1080p, and a
# 40-megapixel phone photo would only cost memory.
MAX_IMAGE_SIDE = 2048
# 15 MiB of encoded bytes. app.py reads this same constant to stop an upload
# early, so the HTTP limit and this check cannot drift apart.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Checked from the header, before any pixel is decoded: a few kilobytes of
# PNG can declare a gigapixel canvas.
MAX_IMAGE_PIXELS = 40_000_000

# Text chunks (PNG tEXt, JPEG comment, XMP) that can carry a rights notice.
_INFO_NOTICE_KEYS = ("copyright", "author", "artist", "description", "comment", "rights")
# EXIF is organised as numbered tags grouped into IFDs ("image file
# directories"). 0x8769 points at the Exif sub-IFD, a second directory that
# holds camera-specific tags; 0x9286 (UserComment) lives there, not at the top.
_EXIF_SUB_IFD = 0x8769
_EXIF_USER_COMMENT = 0x9286

# EXIF tags that can carry rights or usage text a human should read.
# The keys are the standard EXIF tag numbers; the values are our labels.
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
        """Keep the full report so the API can return every issue, not just the text."""
        self.report = report
        # The message joins every error so a plain str(err) is still useful;
        # the fallback covers a failed report that listed no errors.
        super().__init__(
            "; ".join(issue.message for issue in report.errors)
            or "photo rejected by the quality gate"
        )


# @dataclass writes __init__, __repr__ and __eq__ from the annotated fields,
# so a record is just its data with no boilerplate.
@dataclass
class AvatarRecord:
    """One registered (or merely present) face image."""

    avatar_id: str
    path: Path
    width: int
    height: int
    # False when the sidecar is missing or forbids use; usability_reason says
    # which, in words the UI can show.
    usable: bool
    usability_reason: str
    # default_factory builds a fresh dict per record. A plain ``= {}`` default
    # would be one dict shared by every instance, which dataclass refuses.
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """The record as the camelCase JSON the API returns."""
        return {
            "avatarId": self.avatar_id,
            # Only the file name, never the absolute path: the server's
            # directory layout is not the client's business.
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

    Raises ``AvatarError`` when the id does not match ``AVATAR_ID_PATTERN``.
    """
    # The isinstance check comes first: a JSON body can carry a number or
    # null, and re.match on a non-string would raise TypeError instead.
    if not isinstance(avatar_id, str) or not AVATAR_ID_PATTERN.match(avatar_id):
        raise AvatarError(
            "avatarId must be 1-64 characters of letters, digits, '_' or '-', "
            f"starting with a letter or digit (got {avatar_id!r})"
        )
    return avatar_id


def _embedded_notices(image) -> Dict[str, str]:
    """
    Rights-related text found in the image's metadata, if any.

    Looks in three places: EXIF tags, the Exif sub-IFD's UserComment, and the
    format's own text chunks (PNG tEXt, JPEG comments, XMP). Returns a dict of
    label -> text; empty when nothing was found. Never raises: unreadable
    metadata only means there is nothing to report.
    """
    notices: Dict[str, str] = {}
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - a malformed EXIF block is not fatal
        return notices

    # A nested helper: it closes over ``notices`` so each source below adds
    # to the same dict with the same cleaning rules.
    def keep(label: str, value) -> None:
        # EXIF strings often arrive as raw bytes; decode leniently so one bad
        # byte does not hide the rest of a notice.
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        # Many cameras pad fields with NUL bytes or spaces; a value that is
        # only padding is not a notice.
        if isinstance(value, str) and value.strip("\x00 \t\r\n"):
            # setdefault keeps the first source that supplied a label, and the
            # 500-character cap stops a huge field from bloating the sidecar.
            notices.setdefault(label, value.strip("\x00 \t\r\n")[:500])

    for tag, label in _EXIF_NOTICE_TAGS.items():
        keep(label, exif.get(tag))
    # UserComment lives in the Exif sub-IFD, behind an 8-byte charset marker.
    try:
        comment = exif.get_ifd(_EXIF_SUB_IFD).get(_EXIF_USER_COMMENT)
    except Exception:  # noqa: BLE001
        comment = None
    if isinstance(comment, bytes) and len(comment) > 8:
        # Drop the marker (e.g. b"ASCII\0\0\0") so only the text remains.
        comment = comment[8:]
    keep("userComment", comment)
    # Rights text outside EXIF: PNG text chunks, JPEG comments, XMP.
    # Pillow exposes those as the ``image.info`` dict; getattr guards against
    # an object that has no ``info`` at all.
    for key, value in getattr(image, "info", {}).items():
        lowered = str(key).lower()
        if any(name in lowered for name in _INFO_NOTICE_KEYS):
            keep(str(key), value)
        elif lowered in ("xmp", "xml:com.adobe.xmp") and value:
            # XMP is an XML document; the rights statement sits in the Dublin
            # Core <dc:rights> element. re.S lets ".*?" span line breaks.
            text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
            match = re.search(r"<dc:rights>(.*?)</dc:rights>", text, re.S)
            if match:
                # Strip the inner XML tags (rdf:Alt, rdf:li, ...) to plain text.
                keep("xmpRights", re.sub(r"<[^>]+>", " ", match.group(1)))
    return notices


def decode_image(image: Union[bytes, bytearray, str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Decode to an upright RGB uint8 array, capped at ``MAX_IMAGE_SIDE``.

    Returns the array and any rights notices embedded in the file.

    Accepts raw upload bytes, a file path, or an array already in memory.
    Raises ``AvatarError`` when the input is too large, missing, not an image,
    corrupt, or an array of an unsupported shape.
    """
    # Imported here, not at the top, so importing this module (for example in
    # a test that only needs validate_avatar_id) does not pay for Pillow.
    from PIL import Image, ImageOps, UnidentifiedImageError

    if isinstance(image, np.ndarray):
        array = image
        # Normalise the array to height x width x 3 channels:
        if array.ndim == 2:
            # Greyscale: repeat the one channel three times to make RGB.
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] == 4:
            # RGBA: drop the alpha (transparency) channel.
            array = array[:, :, :3]
        elif array.ndim != 3 or array.shape[2] != 3:
            raise AvatarError(f"unsupported image array shape {image.shape}")
        # Clip before the cast: astype(uint8) on 300 would wrap to 44, not
        # saturate at 255.
        handle = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        # An array has no file metadata, so there are no notices to read.
        notices: Dict[str, str] = {}
    else:
        try:
            if isinstance(image, (bytes, bytearray)):
                # Size check on the encoded bytes first: the cheapest refusal.
                if len(image) > MAX_UPLOAD_BYTES:
                    raise AvatarError(
                        f"image is {len(image) / 1e6:.1f} MB; the limit is "
                        f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB"
                    )
                # BytesIO wraps the bytes in a file-like object Pillow can read.
                handle = Image.open(io.BytesIO(bytes(image)))
            else:
                path = Path(image)
                if not path.is_file():
                    raise AvatarError(f"image not found: {path}")
                handle = Image.open(path)
            # Image.open is lazy: it has read only the header so far, which is
            # what makes this dimension check safe before decoding.
            if handle.width * handle.height > MAX_IMAGE_PIXELS:
                raise AvatarError(
                    f"image is {handle.width}x{handle.height} pixels; the limit is "
                    f"{MAX_IMAGE_PIXELS // 1_000_000} megapixels"
                )
            # Force the full decode now, inside the try, so a truncated file
            # fails here with a clear AvatarError rather than later.
            handle.load()
        except UnidentifiedImageError as err:
            raise AvatarError(
                "the file is not an image this project can read "
                f"(supported: {', '.join(IMAGE_EXTENSIONS)})"
            ) from err
        except (OSError, Image.DecompressionBombError) as err:
            # Truncated or corrupt data, or a canvas PIL itself refuses.
            raise AvatarError(f"the image could not be decoded: {err}") from err
        # Notices are read before exif_transpose/convert, which return a new
        # image and can drop the metadata we want to inspect.
        notices = _embedded_notices(handle)
        # exif_transpose only returns None when asked to work in place, which
        # we never do; the fallback keeps the type honest without a branch.
        # It rotates the pixels to match the EXIF orientation tag, so a phone
        # portrait is upright once the tag is gone.
        handle = (ImageOps.exif_transpose(handle) or handle).convert("RGB")

    longest = max(handle.size)
    if longest > MAX_IMAGE_SIDE:
        # One scale factor for both sides keeps the aspect ratio; max(1, ...)
        # stops a very thin image from rounding a side down to zero.
        scale = MAX_IMAGE_SIDE / longest
        handle = handle.resize(
            (max(1, round(handle.width * scale)), max(1, round(handle.height * scale))),
            # Resampling.LANCZOS is the canonical name since Pillow 9.1; the
            # bare Image.LANCZOS is a legacy alias.
            Image.Resampling.LANCZOS,
        )
    # ascontiguousarray guarantees one unbroken block of memory, the layout
    # native image libraries read without an extra copy.
    return np.ascontiguousarray(np.asarray(handle, dtype=np.uint8)), notices


class AvatarStore:
    """
    Filesystem-backed registry of avatar images and their provenance.

    The folder is the database: each avatar is ``<root>/<avatarId>.<ext>``
    plus its provenance sidecar. There is no index file to fall out of sync.
    """

    def __init__(
        self,
        root: Optional[Path | str] = None,
        engine: Optional[FaceMeshEngine] = None,
    ) -> None:
        """
        ``root`` defaults to ``inputs/faces/``; tests pass a temporary folder.

        ``engine`` is the face-landmark engine used by the quality gate. Left
        as None, the process-wide shared engine is used, loaded on first need.
        Tests inject a fake one so no model is downloaded.
        """
        self.root = Path(root) if root is not None else FACES_DIR
        self._engine = engine

    # -- helpers ------------------------------------------------------------

    def _engine_or_shared(self) -> FaceMeshEngine:
        # Resolved at call time, not in __init__, so creating a store (which
        # app.py does at import) never loads MediaPipe.
        return self._engine if self._engine is not None else shared_face_engine()

    def _find(self, avatar_id: str) -> Optional[Path]:
        # Same rule as ``list``: the extension is matched case-insensitively,
        # so whatever is listed can also be fetched and deleted.
        if not self.root.is_dir():
            return None
        # sorted() makes the choice deterministic when two files share a stem
        # (say x.jpg and x.png): the same one wins here and in ``list``.
        for path in sorted(self.root.iterdir()):
            if path.stem == avatar_id and path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                return path
        return None

    def _record(self, avatar_id: str, path: Path) -> AvatarRecord:
        # Build the record for one file: its size, and what its sidecar allows.
        width = height = 0
        try:
            from PIL import Image

            # Opening reads only the header, so getting the size is cheap even
            # for a large photo.
            with Image.open(path) as handle:
                width, height = handle.size
        except Exception as err:  # noqa: BLE001
            # An unreadable file is still listed (with 0x0) so the user can see
            # and delete it, rather than having it vanish from the listing.
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
        # Tracks ids already listed so x.jpg and x.png produce one entry.
        seen = set()
        for path in sorted(self.root.iterdir()):
            # Skips sidecars (.json), sub-folders and anything else that is
            # not an image.
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            avatar_id = path.stem
            # A hand-copied file named "my photo.jpg" has no valid id; listing
            # it would offer an id that ``get`` then refuses.
            if avatar_id in seen or not AVATAR_ID_PATTERN.match(avatar_id):
                continue
            seen.add(avatar_id)
            records.append(self._record(avatar_id, path))
        return records

    def get(self, avatar_id: str) -> AvatarRecord:
        """
        The record for one avatar, whether or not it may be used.

        Raises ``AvatarError`` for a malformed id and ``AvatarNotFound`` when
        no image has that id. Use ``require_usable`` before rendering.
        """
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
        """
        ``get``, but refuse an image whose provenance does not permit use.

        Raises ``AvatarConsentError`` (on top of what ``get`` raises) when the
        sidecar is missing or does not allow the face to be animated.
        """
        record = self.get(avatar_id)
        if not record.usable:
            raise AvatarConsentError(
                f"avatar {avatar_id!r} cannot be used: {record.usability_reason}"
            )
        return record

    def load_image(self, avatar_id: str) -> np.ndarray:
        """
        The avatar as an RGB array, after the consent check.

        This is the only way the renderers read avatar pixels, so a face
        without consent never reaches a model.
        """
        record = self.require_usable(avatar_id)
        from PIL import Image

        # The file was written by ``register`` as PNG, so no EXIF rotation or
        # size cap is needed here; convert("RGB") still guards a hand-placed
        # file in another mode.
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

        Returns the new record and the quality report (None when
        ``check_quality`` is False, which skips the face model entirely).
        """
        # Cheap checks first, in order of cost: id, consent, existence, then
        # decoding, then the face model. A bad request fails before any heavy
        # work starts.
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
            # The face engine is shared by every request in the process and is
            # not documented as thread-safe, so its use is serialised with one
            # lock (see face_engine.FACE_ENGINE_LOCK).
            with FACE_ENGINE_LOCK:
                report = self._engine_or_shared().check_quality(array)
            if not report.passed:
                raise AvatarRejected(report)

        from PIL import Image

        self.root.mkdir(parents=True, exist_ok=True)
        # Always stored as PNG whatever came in: lossless, and re-encoding is
        # what drops the EXIF block.
        path = self.root / f"{avatar_id}.png"
        # Fail closed: the old record goes before the new pixels arrive. If
        # anything below fails, the image has no sidecar and cannot be used,
        # instead of a new face inheriting the previous subject's consent.
        if existing is not None:
            provenance.sidecar_path(existing).unlink(missing_ok=True)
            # An old x.jpg would otherwise sit beside the new x.png, and
            # ``_find`` could return the stale one.
            if existing != path:
                existing.unlink()
        # Also clear a stray sidecar left for this exact path.
        provenance.sidecar_path(path).unlink(missing_ok=True)
        Image.fromarray(array, "RGB").save(path, format="PNG")

        # Copy so the caller's ``extra`` dict is not mutated below.
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
        # The sidecar is written last: only once it exists is the avatar usable.
        # ``speaker`` is the provenance field name shared with voice clips.
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
        """
        Remove an avatar and its provenance record.

        Raises ``AvatarNotFound`` when there is nothing to delete. The image
        goes first, so a failure in between leaves an orphan sidecar (which
        ``list`` ignores) rather than an unlabelled image.
        """
        record = self.get(avatar_id)
        record.path.unlink(missing_ok=True)
        provenance.sidecar_path(record.path).unlink(missing_ok=True)
        logger.info("Deleted avatar %s", avatar_id)
