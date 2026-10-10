"""Tests for API-key auth and rate limiting."""

import unittest

from security import API_KEY_HEADER, SecurityConfig, SecurityGate, TokenBucketLimiter


def gate(
    auth_enabled: bool = True,
    keys=("secret-key-a", "secret-key-b"),
    rate_limit_enabled: bool = False,
    rpm: int = 120,
    burst: int = 10,
) -> SecurityGate:
    return SecurityGate(
        SecurityConfig(
            auth_enabled=auth_enabled,
            api_keys=frozenset(keys),
            rate_limit_enabled=rate_limit_enabled,
            requests_per_minute=rpm,
            burst=burst,
        )
    )


class AuthenticationTests(unittest.TestCase):
    def test_valid_key_is_accepted(self):
        allowed, code, _ = gate().inspect("/api/v1/audio/synthesize", "secret-key-a", "1.2.3.4")
        self.assertTrue(allowed)
        self.assertEqual(code, 0)

    def test_missing_key_is_rejected_with_401(self):
        allowed, code, detail = gate().inspect("/api/v1/audio/synthesize", None, "1.2.3.4")
        self.assertFalse(allowed)
        self.assertEqual(code, 401)
        self.assertEqual(detail["headers"]["WWW-Authenticate"], API_KEY_HEADER)

    def test_wrong_key_is_rejected(self):
        allowed, code, _ = gate().inspect("/api/v1/audio/synthesize", "nope", "1.2.3.4")
        self.assertFalse(allowed)
        self.assertEqual(code, 401)

    def test_auth_disabled_lets_everything_through(self):
        allowed, _, _ = gate(auth_enabled=False, keys=()).inspect(
            "/api/v1/audio/synthesize", None, "1.2.3.4"
        )
        self.assertTrue(allowed)

    def test_public_paths_never_need_a_key(self):
        secured = gate()
        for path in ("/health", "/docs", "/openapi.json", "/outputs/speech.wav"):
            with self.subTest(path=path):
                allowed, _, _ = secured.inspect(path, None, "1.2.3.4")
                self.assertTrue(allowed)

    def test_auth_enabled_with_no_keys_rejects_everything(self):
        """A misconfigured deployment must fail closed, not open."""
        allowed, code, _ = gate(keys=()).inspect("/api/v1/audio/synthesize", "any", "1.2.3.4")
        self.assertFalse(allowed)
        self.assertEqual(code, 401)

    def test_describe_never_leaks_key_material(self):
        summary = gate().describe()
        self.assertNotIn("secret-key-a", str(summary))
        self.assertEqual(summary["configuredKeys"], 2)


class RateLimitTests(unittest.TestCase):
    def test_burst_is_allowed_then_throttled(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=5)
        for index in range(5):
            allowed, _ = limiter.check("client", now=100.0)
            self.assertTrue(allowed, f"request {index} should be allowed")
        allowed, retry_after = limiter.check("client", now=100.0)
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0.0)

    def test_tokens_refill_over_time(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=1)
        self.assertTrue(limiter.check("client", now=0.0)[0])
        self.assertFalse(limiter.check("client", now=0.0)[0])
        # 60 rpm is one token per second.
        self.assertTrue(limiter.check("client", now=1.0)[0])

    def test_identities_have_separate_budgets(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=1)
        self.assertTrue(limiter.check("a", now=0.0)[0])
        self.assertTrue(limiter.check("b", now=0.0)[0])
        self.assertFalse(limiter.check("a", now=0.0)[0])

    def test_refill_never_exceeds_capacity(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=3)
        limiter.check("client", now=0.0)
        # A long idle period must not bank unlimited tokens.
        allowed_count = 0
        for _ in range(10):
            if limiter.check("client", now=10_000.0)[0]:
                allowed_count += 1
        self.assertEqual(allowed_count, 3)

    def test_gate_returns_429_with_retry_after(self):
        secured = gate(rate_limit_enabled=True, rpm=60, burst=1)
        self.assertTrue(secured.inspect("/api/v1/x", "secret-key-a", "ip")[0])
        allowed, code, detail = secured.inspect("/api/v1/x", "secret-key-a", "ip")
        self.assertFalse(allowed)
        self.assertEqual(code, 429)
        self.assertIn("Retry-After", detail["headers"])

    def test_rate_limit_keys_on_the_api_key_not_the_ip(self):
        secured = gate(rate_limit_enabled=True, rpm=60, burst=1)
        self.assertTrue(secured.inspect("/api/v1/x", "secret-key-a", "ip-1")[0])
        # Same key from a different IP shares the budget.
        self.assertFalse(secured.inspect("/api/v1/x", "secret-key-a", "ip-2")[0])
        # A different key gets its own.
        self.assertTrue(secured.inspect("/api/v1/x", "secret-key-b", "ip-1")[0])

    def test_identity_does_not_embed_the_whole_key(self):
        identity = gate().identity("super-secret-value", "1.2.3.4")
        self.assertNotIn("super-secret-value", identity)


class ConfigTests(unittest.TestCase):
    def test_config_parses_comma_separated_keys(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"API_KEYS": "a, b ,, c", "AUTH_ENABLED": "true"}):
            config = SecurityConfig.from_env()
        self.assertEqual(config.api_keys, frozenset({"a", "b", "c"}))
        self.assertTrue(config.auth_enabled)

    def test_defaults_keep_local_development_open(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            config = SecurityConfig.from_env()
        self.assertFalse(config.auth_enabled)
        self.assertTrue(config.rate_limit_enabled)


if __name__ == "__main__":
    unittest.main()


class RateLimitIdentityTests(unittest.TestCase):
    """Found in the T6.5 review: with auth off, a rotating X-API-Key beat the limiter."""

    def test_with_auth_off_a_fresh_random_key_each_request_does_not_escape_the_limit(self):
        g = gate(auth_enabled=False, keys=(), rate_limit_enabled=True, rpm=60, burst=5)
        allowed = sum(g.inspect("/api/v1/audio/languages", f"random-{i}", "9.9.9.9")[0] for i in range(200))
        self.assertEqual(allowed, 5)  # the burst, exactly as for a client with no key
        self.assertEqual(len(g.limiter._buckets), 1)  # and one bucket, not 200

    def test_with_auth_on_each_valid_key_still_has_its_own_bucket(self):
        g = gate(auth_enabled=True, rate_limit_enabled=True, rpm=60, burst=3)
        for _ in range(3):
            self.assertTrue(g.inspect("/api/v1/audio/languages", "secret-key-a", "9.9.9.9")[0])
        self.assertFalse(g.inspect("/api/v1/audio/languages", "secret-key-a", "9.9.9.9")[0])
        self.assertTrue(g.inspect("/api/v1/audio/languages", "secret-key-b", "9.9.9.9")[0])

    def test_an_unauthenticated_flood_cannot_drain_a_valid_keys_bucket(self):
        g = gate(auth_enabled=True, rate_limit_enabled=True, rpm=60, burst=3)
        for _ in range(500):
            self.assertEqual(g.inspect("/api/v1/audio/languages", "wrong", "9.9.9.9")[1], 401)
        self.assertTrue(g.inspect("/api/v1/audio/languages", "secret-key-a", "9.9.9.9")[0])
        self.assertEqual(len(g.limiter._buckets), 1)  # only the valid key's bucket exists

    def test_idle_full_buckets_are_forgotten_so_memory_stays_bounded(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=2)
        limiter.MAX_BUCKETS = 50
        for i in range(200):
            limiter.check(f"ip:{i}", now=0.0)          # each spends one of two tokens
        limiter.check("ip:late", now=1000.0)           # long after: every earlier bucket has refilled
        self.assertLess(len(limiter._buckets), 51)
        self.assertIn("ip:late", limiter._buckets)

    def test_pruning_never_forgets_a_client_who_is_still_throttled(self):
        limiter = TokenBucketLimiter(requests_per_minute=60, burst=2)
        limiter.MAX_BUCKETS = 5
        for _ in range(2):
            limiter.check("busy", now=0.0)
        for i in range(20):
            limiter.check(f"idle:{i}", now=0.0)
        self.assertFalse(limiter.check("busy", now=0.1)[0])  # still empty: its bucket survived the prune
