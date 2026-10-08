"""
API authentication and rate limiting.

The assignment's Milestone 1 acceptance matrix requires both before the API can
be called public-ready, and Phase 5 later builds abuse detection on top of the
same counters.

Design notes:
  * Auth is an API-key header check, off by default so local development and
    the existing test suite keep working. Set ``AUTH_ENABLED=true`` and
    ``API_KEYS=key1,key2`` in ``.env`` to turn it on.
  * Keys are compared with ``hmac.compare_digest`` to avoid leaking a key
    through response timing.
  * The limiter is a per-identity token bucket held in process memory, which is
    correct for the single-process dev server. Known limitation: with several
    API workers each process keeps its own buckets, so the effective limit is
    ``RATE_LIMIT_RPM x worker_count``. A shared Redis bucket is the Phase 5
    follow-up, tracked in docs/context.md.

**How to say this in an interview:** "Keys are compared in constant time to
avoid a timing oracle, and throttling uses a token bucket so short bursts are
absorbed while the sustained rate is capped. The buckets are per-process, which
is a documented limitation rather than an accident."
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Named once rather than typed as a literal at each use, so the header appears
# in the 401 body, the WWW-Authenticate header and /health without drifting.
API_KEY_HEADER = "X-API-Key"

# Paths that never require a key, so health checks and docs stay reachable.
# A set, not a list: membership testing is O(1) and this runs on every request.
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}
# A tuple because str.startswith() accepts one and tests them all in a single
# call. The trailing slash matters: "/outputs/" cannot be matched by a path
# like "/outputs-private", which "/outputs" would have allowed through.
PUBLIC_PREFIXES = ("/outputs/",)


def _env_bool(name: str, default: bool = False) -> bool:
    """
    Parse a boolean from the environment, where everything is a string.

    Needed because ``bool(os.getenv("AUTH_ENABLED"))`` is True for the string
    "false" - a classic config bug that would silently enable a feature.
    """
    raw = os.getenv(name)
    # `is None` distinguishes "unset" from "set to empty string". An explicit
    # empty value falls through and reads as False, which is the safer reading.
    if raw is None:
        return default
    # Accepts the spellings people actually write in a .env file. Anything else
    # is False, so a typo fails closed for a feature flag.
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an int from the environment, falling back rather than crashing."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        # A malformed RATE_LIMIT_RPM should not stop the server booting; it
        # should run with the documented default. Note it still rate-limits -
        # failing *open* here would be the dangerous choice.
        return default


@dataclass(frozen=True)
class SecurityConfig:
    """
    The security settings, resolved once at startup.

    ``frozen=True`` makes it immutable, which is what you want for a policy
    object: nothing can quietly raise its own rate limit at runtime, and it is
    safe to read from many threads without a lock.
    """

    auth_enabled: bool
    # frozenset, not set: immutable to match the frozen dataclass, and hashable.
    api_keys: frozenset[str]
    rate_limit_enabled: bool
    requests_per_minute: int
    burst: int

    # A classmethod used as an alternative constructor - `cls` is the class, so
    # `cls(...)` builds an instance. Separating "read the environment" from
    # "hold the settings" is what lets the tests construct a SecurityConfig
    # directly with no environment at all.
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        raw_keys = os.getenv("API_KEYS", "")
        # Split on comma, strip whitespace, and drop empties - so "a, b," and
        # "a,b" parse identically and a trailing comma cannot create an empty
        # key that would then match a missing header.
        keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
        rpm = _env_int("RATE_LIMIT_RPM", 120)
        return cls(
            auth_enabled=_env_bool("AUTH_ENABLED", False),
            api_keys=keys,
            # Rate limiting defaults to ON while auth defaults to OFF: an
            # unauthenticated dev server is a convenience, an unthrottled one
            # is a way to melt the GPU with a stuck polling loop.
            rate_limit_enabled=_env_bool("RATE_LIMIT_ENABLED", True),
            # max(1, ...) keeps a nonsensical 0 or negative from disabling the
            # limiter by making every bucket permanently empty or infinite.
            requests_per_minute=max(1, rpm),
            # Allow a short burst so a UI that fires several polls at once is
            # not throttled, while the sustained rate stays at the limit.
            burst=max(1, _env_int("RATE_LIMIT_BURST", max(10, rpm // 4))),
        )


class TokenBucketLimiter:
    """
    Per-identity token bucket.

    Each identity accrues ``rate`` tokens per second up to ``capacity``. A
    request costs one token; when the bucket is empty the request is rejected
    with the number of seconds until the next token.

    Why a token bucket rather than a fixed window ("120 requests per calendar
    minute"): a fixed window allows 240 requests across a window boundary - 120
    at 11:59:59 and 120 at 12:00:00 - and resets in a thundering herd. A bucket
    refills continuously, so the sustained rate is a real ceiling while a short
    burst up to ``capacity`` is still absorbed.

    **How to say this in an interview:** "Token bucket: continuous refill at
    the sustained rate, with a capacity that defines the allowed burst. Avoids
    the boundary doubling and the synchronised reset of a fixed window."
    """

    def __init__(self, requests_per_minute: int, burst: int):
        # Per-second refill, because the bucket is evaluated at request time
        # rather than on a timer. Storing it as a rate means no background task
        # is needed to "add tokens" - they are computed on read.
        self.rate = requests_per_minute / 60.0
        self.capacity = float(max(burst, 1))
        # identity -> (tokens_remaining, timestamp_of_last_update). A tuple
        # rather than an object: two floats per identity keeps this cheap even
        # with many clients. Unbounded, which is the memory caveat - an
        # attacker rotating IPs grows this dict, so it is pruned past MAX_BUCKETS.
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    # Past this many identities, buckets that have refilled completely are
    # dropped: a full bucket is exactly what a new identity starts with, so
    # forgetting it changes no decision.
    MAX_BUCKETS = 10_000

    def _prune(self, now: float) -> None:
        if len(self._buckets) <= self.MAX_BUCKETS:
            return
        self._buckets = {
            identity: (tokens, last)
            for identity, (tokens, last) in self._buckets.items()
            if tokens + (now - last) * self.rate < self.capacity
        }

    def check(self, identity: str, now: Optional[float] = None) -> Tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        # time.monotonic(), never time.time(): monotonic cannot jump backwards
        # when the clock is adjusted or NTP steps it. With wall-clock time, a
        # backwards step would make (now - last) negative and *remove* tokens.
        # The `now` parameter is injected by tests so they need no sleeps.
        now = time.monotonic() if now is None else now
        # The whole read-modify-write must be atomic: two threads could
        # otherwise both read the last token and both be allowed through.
        with self._lock:
            self._prune(now)
            # An unseen identity starts full, so a first request is never
            # throttled. `now` as the default timestamp means its first refill
            # calculation adds nothing rather than a huge backdated credit.
            tokens, last = self._buckets.get(identity, (self.capacity, now))
            # The refill: elapsed seconds x rate, clamped to capacity. The
            # min() is what stops an idle client banking unlimited quota.
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                # Spend one token and record the time it was spent.
                self._buckets[identity] = (tokens - 1.0, now)
                return True, 0.0
            # Rejected - but still write back, so the partial refill is not
            # lost and `last` advances. Skipping this write would re-credit the
            # same elapsed time on the next call.
            self._buckets[identity] = (tokens, now)
            # How long until the bucket reaches one whole token. The rate > 0
            # guard avoids a ZeroDivisionError; 60 s is a safe stand-in.
            retry_after = (1.0 - tokens) / self.rate if self.rate > 0 else 60.0
            return False, retry_after

    def reset(self, identity: Optional[str] = None) -> None:
        """Clear one identity's bucket, or all of them. Used by the tests."""
        with self._lock:
            if identity is None:
                self._buckets.clear()
            else:
                # pop with a default never raises on an unknown identity.
                self._buckets.pop(identity, None)

    def tracked_identities(self) -> int:
        """How many buckets are held - the memory caveat, observable."""
        with self._lock:
            return len(self._buckets)


class SecurityGate:
    """Combines key auth and rate limiting behind one ``inspect`` call."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        # Injectable config again, so tests build a gate with exact settings.
        self.config = config or SecurityConfig.from_env()
        self.limiter = TokenBucketLimiter(
            self.config.requests_per_minute, self.config.burst
        )
        # Fail loudly at startup rather than mysteriously at request time: this
        # combination rejects every call, and the operator almost certainly did
        # not mean it. logger.error, because it is a misconfiguration.
        if self.config.auth_enabled and not self.config.api_keys:
            logger.error(
                "AUTH_ENABLED=true but API_KEYS is empty - every request will be "
                "rejected. Set API_KEYS in .env."
            )

    # ------------------------------------------------------------------

    @staticmethod
    def is_public(path: str) -> bool:
        """Whether this path bypasses auth entirely."""
        # Exact match against the set, or any of the prefixes.
        return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)

    def verify_key(self, presented: Optional[str]) -> bool:
        """Constant-time comparison of the presented key against the allow-list."""
        # Auth off = everything verifies. Checked first so development pays
        # nothing for this code path.
        if not self.config.auth_enabled:
            return True
        if not presented:
            return False
        # hmac.compare_digest, not ==. A normal string comparison returns as
        # soon as two characters differ, so the time it takes leaks how many
        # leading characters were correct; an attacker can then guess a key one
        # character at a time. compare_digest always examines the whole value.
        #
        # any() does short-circuit across the *list of keys*, which is fine:
        # that leaks how many keys are configured, not what they are.
        return any(
            hmac.compare_digest(presented, known) for known in self.config.api_keys
        )

    def identity(self, api_key: Optional[str], client_host: Optional[str]) -> str:
        """
        Rate-limit bucket key: the API key when present, else the client IP.

        The key is hashed rather than truncated. A prefix would let two distinct
        keys that happen to share their first characters land in the same
        bucket and consume each other's quota.
        """
        if api_key:
            # Hashing also keeps the secret out of the bucket dict, so a memory
            # dump or a debug log of _buckets cannot expose live API keys.
            digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
            # 32 hex chars = 128 bits, far beyond collision risk here, and the
            # "key:" / "ip:" prefixes stop an IP ever colliding with a hash.
            return f"key:{digest[:32]}"
        return f"ip:{client_host or 'unknown'}"

    def inspect(
        self,
        path: str,
        api_key: Optional[str],
        client_host: Optional[str],
    ) -> Tuple[bool, int, dict]:
        """
        Evaluate one request.

        Returns ``(allowed, status_code, detail)``. ``status_code`` is 0 when
        allowed; ``detail`` carries the response body and any headers to add.

        Returning data rather than raising an HTTPException keeps this module
        framework-agnostic and directly unit-testable: the tests call inspect()
        with plain strings, no HTTP client involved.
        """
        if self.is_public(path):
            return True, 0, {}

        # Order matters: authenticate before throttling. Rate-limiting first
        # would let an unauthenticated flood consume the bucket belonging to
        # the IP, which is a denial-of-service against legitimate users there.
        if not self.verify_key(api_key):
            return False, 401, {
                # Golden rule 7: the error names the fix, including the header.
                "detail": (
                    f"Missing or invalid API key. Send it in the {API_KEY_HEADER} header."
                ),
                # The standard challenge header for a 401.
                "headers": {"WWW-Authenticate": API_KEY_HEADER},
            }

        if self.config.rate_limit_enabled:
            # The presented key only names a bucket when auth is on, i.e. when
            # it has been verified above. With auth off it is whatever the
            # client typed: honouring it let a caller send a fresh random
            # X-API-Key on every request, get a fresh bucket each time, and
            # never be limited (200 of 200 requests allowed, measured).
            allowed, retry_after = self.limiter.check(
                self.identity(api_key if self.config.auth_enabled else None, client_host)
            )
            if not allowed:
                return False, 429, {
                    "detail": (
                        f"Rate limit exceeded: {self.config.requests_per_minute} "
                        "requests/minute."
                    ),
                    # Retry-After must be whole seconds. +0.5 then int() rounds
                    # to nearest instead of truncating, and max(1, ...) avoids
                    # telling a client to retry after 0 seconds (an instant
                    # retry that would just be rejected again).
                    "headers": {"Retry-After": str(max(1, int(retry_after + 0.5)))},
                }

        return True, 0, {}

    def describe(self) -> dict:
        """Non-secret summary for /health and the UI."""
        # Deliberately reports the *count* of configured keys, never the keys.
        # /health is a public path, so anything returned here is world-readable.
        return {
            "authEnabled": self.config.auth_enabled,
            "configuredKeys": len(self.config.api_keys),
            "rateLimitEnabled": self.config.rate_limit_enabled,
            "requestsPerMinute": self.config.requests_per_minute,
            "burst": self.config.burst,
            # Published so the frontend reads the header name from the server
            # instead of hardcoding it.
            "apiKeyHeader": API_KEY_HEADER,
        }
