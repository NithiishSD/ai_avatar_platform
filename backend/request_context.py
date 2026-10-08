"""
Request ids and structured log lines (R-27).

Every HTTP request gets an id: the caller's ``X-Request-ID`` when it is a safe
short token, otherwise a fresh one. The id is returned in the response header
and stamped on every log line written while the request is handled, so one
failing request can be followed through the API, the queue and the engines
with a single ``grep``.

How the id reaches log lines: it lives in a ``ContextVar``, Python's
per-task / per-thread storage. FastAPI runs ``def`` endpoints on a worker
thread but copies the context into it, so the variable is still set there. A
log-record factory then adds ``record.request_id`` to *every* record from
*every* logger, including libraries, with no change to any ``logger.info``
call. Work handed to a background thread pool does not inherit the context by
itself; ``run_in_context`` carries it across.
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# The caller controls this value and it is written into logs, so only a short
# token of safe characters is accepted: no spaces or newlines (log forging).
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
NO_REQUEST = "-"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default=NO_REQUEST)

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


def current_request_id() -> str:
    return _request_id.get()


def new_request_id(incoming: str | None) -> str:
    """The caller's id if it is safe, else a fresh 16-hex-digit one."""
    if incoming and _SAFE_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


def run_in_context(fn: Callable[..., object]) -> Callable[..., object]:
    """Wrap ``fn`` so a thread-pool worker runs it with the *caller's* request id."""
    context = contextvars.copy_context()
    return lambda *args, **kwargs: context.run(fn, *args, **kwargs)


_installed = False


def install_logging() -> None:
    """Stamp ``request_id`` on every log record (import time, so tests see it too). Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    previous_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        record.request_id = _request_id.get()
        return record

    logging.setLogRecordFactory(factory)


def configure_log_output() -> None:
    """Print app logs with the id. Called when the server starts, never on import:
    importing the app in a test must not make the test run noisy."""
    root = logging.getLogger()
    if not root.handlers:  # leave a host application's own logging alone
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def bind_request_id(request_id: str) -> contextvars.Token:
    """Make ``request_id`` current for this task (a WebSocket, which the HTTP middleware never sees)."""
    return _request_id.set(request_id)


def unbind_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


async def request_id_middleware(request: Request, call_next):
    """Assign the id, expose it to the logs for this request, return it in the header."""
    request_id = new_request_id(request.headers.get(REQUEST_ID_HEADER))
    token = _request_id.set(request_id)
    try:
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - turned into a 500 that names the id
            # Starlette would build this 500 outside this middleware, with no
            # header. Answering here means the one response a person most needs
            # to trace says which log lines to look for.
            logger.exception("Unhandled error in %s %s", request.method, request.url.path)
            response = JSONResponse(
                status_code=500,
                content={"detail": f"internal error; quote request id {request_id} when reporting it", "requestId": request_id},
            )
    finally:
        _request_id.reset(token)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response
