"""
SQLite persistence for job records (R-21).

The development queue used to keep every job in a dict, so an API restart
forgot them: a client polling a render got a 404 for a job that had been
running. This store keeps each job as one JSON row in a small SQLite file
(standard library, no server), written on every state change and read back
when the process starts.

What a restart can and cannot recover is decided by the queues that use this
store, not here: a job that never started is run again; a job that was
mid-render is marked FAILED with the reason, because the render itself died
with the process and pretending otherwise would leave it "PROCESSING" forever.

Concepts, explained once:

* **SQLite** is a whole SQL database in one file, driven by the standard
  library's ``sqlite3`` module. No server process, nothing to install.
* **WAL** (write-ahead logging) is a SQLite journal mode: changes go to a side
  log first and are folded into the main file later. Readers keep seeing the
  last committed state while a write is in progress, and a crash mid-write
  cannot leave a half-written database.
* An **upsert** (``INSERT ... ON CONFLICT DO UPDATE``) inserts a row, or
  updates it if one with the same key exists, in a single statement.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "outputs" / "jobs.sqlite"

# ``kind`` separates the record families sharing this file (renders, generation
# jobs, batches, ...), so two families can reuse an id without colliding. The record
# is stored as JSON text: the callers own its shape, the table does not.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    kind    TEXT NOT NULL,
    id      TEXT NOT NULL,
    record  TEXT NOT NULL,
    updated REAL NOT NULL DEFAULT (julianday('now')),
    PRIMARY KEY (kind, id)
)
"""


def default_path() -> Path:
    """Where the job file lives: ``$JOBS_DB`` if set, else ``outputs/jobs.sqlite``."""
    return Path(os.getenv("JOBS_DB", str(DEFAULT_PATH)))


class JobStore:
    """Thread-safe key-value store of job records, keyed by ``(kind, id)``."""
    def __init__(self, path: Optional[Path | str] = None) -> None:
        # ":memory:" is SQLite's own name for a throwaway database; the unit
        # tests use it so importing the app never touches the real job file.
        self.path = Path(path) if path is not None else default_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the API threads and the worker thread, so
        # every use takes the lock (sqlite3 objects are not safe to share otherwise).
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL: a crash mid-write leaves the last committed state, not a torn file.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()

    def put(self, kind: str, job_id: str, record: Dict[str, Any]) -> None:
        """Insert or replace the record for ``(kind, job_id)`` and commit at once."""
        with self._lock:
            # ``?`` placeholders let sqlite3 quote the values, so an id can never be read
            # as SQL. ``excluded`` is the row that failed to insert, i.e. the new values.
            self._db.execute(
                "INSERT INTO jobs (kind, id, record) VALUES (?, ?, ?) "
                "ON CONFLICT(kind, id) DO UPDATE SET record = excluded.record, updated = julianday('now')",
                (kind, job_id, json.dumps(record)),
            )
            self._db.commit()

    def get(self, kind: str, job_id: str) -> Optional[Dict[str, Any]]:
        """The stored record, or None when no such job exists."""
        with self._lock:
            row = self._db.execute("SELECT record FROM jobs WHERE kind = ? AND id = ?", (kind, job_id)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Every record of a kind, in the order first submitted, as ``(id, record)``."""
        with self._lock:
            # rowid grows with each new insert and an upsert keeps the original row, so this
            # is submission order even after many updates.
            rows = self._db.execute("SELECT id, record FROM jobs WHERE kind = ? ORDER BY rowid", (kind,)).fetchall()
        return [(job_id, json.loads(record)) for job_id, record in rows]

    def close(self) -> None:
        """Close the connection. The store cannot be used afterwards."""
        with self._lock:
            self._db.close()
