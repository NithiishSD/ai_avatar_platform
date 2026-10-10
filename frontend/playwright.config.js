// End-to-end tests: a real browser against the real backend and the real
// Vite dev server. Nothing is mocked except where a test deliberately delays
// one response to prove the UI handles ordering (see smoke.spec.js).
//
//   cd frontend && npx playwright test
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  // Model loading on CPU is slow; a single test may legitimately take a while.
  timeout: 120_000,
  expect: { timeout: 20_000 },
  // No retries: a flaky test is a bug to fix, and a retry would hide it.
  retries: 0,
  // One worker: the backend keeps one heavy model resident at a time, and
  // tests running in parallel would fight over it (and over port 8000).
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // Playwright starts both servers, waits for them to answer, and stops them
  // afterwards. reuseExistingServer is false on purpose: if an old server is
  // still holding a port, the run fails instead of testing stale code.
  webServer: [
    {
      // The rate limiter (120 requests/minute per client, tested in
      // tests/test_security.py) is lifted for this run: every spec shares one
      // client IP and the render/synthesis polling alone uses ~1 request a
      // second, so a full run otherwise starved the later specs with 429s.
      command: "RATE_LIMIT_RPM=100000 RATE_LIMIT_BURST=100000 PYTHONPATH=. ./.conda/bin/python -m uvicorn app:app --port 8000",
      cwd: "../backend",
      url: "http://localhost:8000/health",
      // The startup weight audit imports torch; give it time on a cold start.
      timeout: 180_000,
      reuseExistingServer: false,
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173",
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
