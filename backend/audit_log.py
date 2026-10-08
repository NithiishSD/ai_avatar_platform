"""
Consent audit trail (R-35): an append-only, tamper-evident record of every use of a voice or a face.

Consent is only worth something if it can be shown afterwards: *this* recording was cloned on
*this* basis, *this* photo was animated on *that* basis, *this* request was refused and why. Each use
appends one row here.

Tamper-evidence. Every row stores the hash of the row before it and its own hash, computed over its
content plus that previous hash (a hash chain, the same idea as a ledger). Editing a row, deleting one
or reordering them breaks every hash after the change, and ``verify_chain`` finds the first broken
link. It does not stop someone with write access to the database from rewriting the whole chain, which
is why the head hash can be published or stored elsewhere (``head`` returns it); what it does is make
quiet, partial edits detectable.

What is recorded is what a reviewer needs and nothing that identifies a person: ids, hashes, the
consent basis, the engine, the request id. Names never enter this table.

Events (the ``event`` column):

    voice_use         a reference recording was cloned            (basis, reference hash, engine, model)
    voice_refused     a clone request was refused                 (reason)
    face_use          an avatar face was animated or streamed     (avatar, source, basis, job / manifest)
    face_refused      an avatar was refused                       (reason)
    face_registered   a photo was registered                      (avatar, source, basis)
    face_generated    a synthetic face was generated              (avatar, seed)
    audio_supplied    a caller's own audio drove an avatar        (audio hash, basis)
    manifest_issued   a signed manifest was issued for a video    (manifest id, video hash, job)
    abuse_alert       a clone request or its output matched a protected voice (kind, similarity, protected id)
    protected_voice_added / protected_voice_removed   the opt-out list changed (id only)

Where it sits: the voice engine, ``app.py``, ``protected_voices.py`` and ``authenticity.py`` call
``shared_audit().record(...)`` at the moment a voice or face is used or refused. ``app.py`` also serves
``query``, ``head`` and ``verify_chain`` so a reviewer can read and check the trail.

Concepts used here:
  * A *hash* (SHA-256) maps any input to a fixed 64-hex-character digest. Changing one byte of the
    input changes the digest completely, and nobody can construct a different input with the same
    digest, so a stored digest pins down the exact content it was computed from.
  * A *hash chain* feeds each row's digest into the next row's digest. Row N's hash therefore
    depends on every row before it, which is what makes a quiet edit visible further down.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from job_store import default_path
from request_context import current_request_id

# The closed list of events. ``record`` refuses anything else, so a typo cannot create a new event
# type that queries and reviewers would never look for.
EVENTS = (
    "voice_use", "voice_refused", "face_use", "face_refused", "face_registered",
    "face_generated", "audio_supplied", "manifest_issued", "abuse_alert", "protected_voice_added",
    "protected_voice_removed",
)
# The "previous hash" of the very first row: 64 zeros, the same length as a SHA-256 hex digest.
GENESIS = "0" * 64

# AUTOINCREMENT makes SQLite never reuse an id, even after the last row is deleted, so a gap in the
# ids is evidence of a deletion (``verify_chain`` relies on that). ``details`` holds a JSON string.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,
    subject    TEXT NOT NULL,
    basis      TEXT,
    request_id TEXT,
    details    TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
)
"""


def _digest(prev_hash: str, ts: str, event: str, subject: str, basis: Optional[str], request_id: Optional[str], details: str) -> str:
    """SHA-256 hex digest of one row's content and the hash of the row before it.

    The fields are serialised as a JSON list, not concatenated, so ("ab", "c") and ("a", "bc") give
    different bytes. Compact separators make the bytes the same on every run.
    """
    body = json.dumps([prev_hash, ts, event, subject, basis, request_id, details], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    """The audit table in a SQLite file (by default the same file as the job store).

    One connection is shared by every thread, guarded by ``_lock``; SQLite connections are not safe
    for concurrent use, and appending must read the last hash and insert as one step.
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False lets request threads share this connection; the lock does the
        # serialising that SQLite's own check would otherwise enforce by refusing.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL (write-ahead log) mode lets readers keep reading while a writer appends.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()

    def record(self, event: str, subject: str, basis: Optional[str] = None, **details: Any) -> Dict[str, Any]:
        """
        Append one entry and return it. ``subject`` names what was used (an avatar id, a recording's
        hash); ``details`` carries the rest. Raises if the entry cannot be written: a use that cannot
        be logged should not silently go ahead unlogged.
        """
        if event not in EVENTS:
            raise ValueError(f"unknown audit event {event!r}; known: {list(EVENTS)}")
        # UTC ISO-8601: sorts as text, so ``query(since=...)`` can compare strings.
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Ties the entry to the HTTP request that caused it, so it can be matched with the server log.
        request_id = current_request_id()
        # sort_keys gives one fixed byte form for the same details, which the hash needs.
        # default=str turns values JSON cannot encode (a Path, a datetime) into text instead of failing.
        body = json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        # Read-last-hash and insert happen under one lock: two writers racing here would both link to
        # the same previous row and fork the chain.
        with self._lock:
            row = self._db.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            digest = _digest(prev, ts, event, subject, basis, request_id, body)
            cursor = self._db.execute(
                "INSERT INTO audit (ts, event, subject, basis, request_id, details, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
                (ts, event, subject, basis, request_id, body, prev, digest),
            )
            self._db.commit()
        return {"id": cursor.lastrowid, "ts": ts, "event": event, "subject": subject, "basis": basis,
                "requestId": request_id, "details": details, "hash": digest}

    @staticmethod
    def _row(row: tuple) -> Dict[str, Any]:
        """A raw SELECT tuple as the API's camelCase dict; ``prev_hash`` (index 7) is left out."""
        return {"id": row[0], "ts": row[1], "event": row[2], "subject": row[3], "basis": row[4],
                "requestId": row[5], "details": json.loads(row[6]), "hash": row[8]}

    def query(self, event: Optional[str] = None, subject: Optional[str] = None, since: Optional[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
        """Newest first. ``since`` is an ISO timestamp.

        Each filter is optional; ``limit`` is clamped to 1..1000 so one call cannot dump the whole table.
        """
        # Only fixed column names are put into the SQL text; every user value goes through a "?"
        # placeholder, which SQLite binds as data, so a filter value cannot inject SQL.
        clauses, params = [], []
        for column, value in (("event", event), ("subject", subject)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, ts, event, subject, basis, request_id, details, prev_hash, hash FROM audit {where} "
                "ORDER BY id DESC LIMIT ?", (*params, max(1, min(int(limit), 1000)))).fetchall()
        return [self._row(r) for r in rows]

    def find_manifest(self, manifest_id: str, max_bit_errors: int = 24) -> Optional[Dict[str, Any]]:
        """
        The ``manifest_issued`` entry whose id is closest to ``manifest_id`` (hex), within ``max_bit_errors``.

        The id is read out of a watermark, and a compressed copy reads it with some bit errors, so an exact
        match would miss exactly the copies that need tracing. 24 wrong bits of 128 is far beyond what
        chance produces between two unrelated random ids (expected ~64), so a nearest match is a match.
        """
        try:
            wanted = int(manifest_id, 16)
        except ValueError:
            # Not hex: it cannot be a manifest id, so there is nothing to find.
            return None
        # Start one past the allowed maximum, so only a match within the limit replaces it.
        best, best_distance = None, max_bit_errors + 1
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, event, subject, basis, request_id, details, prev_hash, hash FROM audit WHERE event = 'manifest_issued'"
            ).fetchall()
        for row in rows:
            entry = self._row(row)
            try:
                # Hamming distance: XOR leaves a 1 wherever the two ids differ; counting the 1s
                # gives the number of differing bits.
                distance = bin(wanted ^ int(entry["subject"], 16)).count("1")
            except ValueError:
                continue
            if distance < best_distance:
                best, best_distance = entry, distance
        return {**best, "bitErrors": best_distance} if best else None

    def head(self) -> Dict[str, Any]:
        """The newest hash and the entry count: store or publish it to make later rewriting detectable."""
        with self._lock:
            row = self._db.execute("SELECT id, hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            count = self._db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
        return {"entries": count, "headHash": row[1] if row else GENESIS, "lastId": row[0] if row else 0}

    def verify_chain(self) -> Dict[str, Any]:
        """Recompute every hash from the start; report the first entry where the chain breaks."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, event, subject, basis, request_id, details, prev_hash, hash FROM audit ORDER BY id"
            ).fetchall()
        # Walk the rows oldest first, recomputing each hash from the previous one.
        prev = GENESIS
        expected_id = None
        for row in rows:
            entry_id, ts, event, subject, basis, request_id, details, prev_hash, digest = row
            if expected_id is not None and entry_id != expected_id:
                return {"valid": False, "entries": len(rows), "brokenAt": entry_id,
                        "reason": f"entry {expected_id} is missing (the log jumps to {entry_id}): a row was deleted"}
            # Two checks: the stored link must equal the hash just computed for the row before, and the
            # row's own hash must still match its content.
            if prev_hash != prev or _digest(prev_hash, ts, event, subject, basis, request_id, details) != digest:
                return {"valid": False, "entries": len(rows), "brokenAt": entry_id,
                        "reason": "this entry's content or its link to the one before it was changed"}
            prev, expected_id = digest, entry_id + 1
        return {"valid": True, "entries": len(rows), "brokenAt": None, "headHash": prev}

    def close(self) -> None:
        """Close the database connection; the log cannot be used afterwards."""
        with self._lock:
            self._db.close()


_shared: Optional[AuditLog] = None
_shared_lock = threading.Lock()


def shared_audit() -> AuditLog:
    """The process-wide audit log, created on first use.

    Locked because two threads arriving together would otherwise each open their own connection.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AuditLog()
        return _shared
