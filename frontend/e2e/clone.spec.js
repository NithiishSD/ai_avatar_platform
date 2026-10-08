// Gate 2, in the browser: a cloned voice drives a rendered avatar video.
// Real OpenVoice V2 weights, real forced alignment, real renderer - the only
// thing missing from the Gate 2 chain is XTTS-v2, which needs the owner's
// licence acceptance (docs/13, M-03), so the cloner under test is OpenVoice.
import { expect, test } from "@playwright/test";

test("a cloned voice becomes a playable lip-synced avatar video", async ({ page }) => {
  // Cold OpenVoice + Kokoro + alignment on a CPU takes about 30 s; allow
  // generously so a slow machine fails on a real fault, not on the clock.
  test.setTimeout(420_000);
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");

  await page.locator("#voice-clone").click();
  await expect(page.locator("#sample-select")).toContainText("ljspeech_reference");
  // selectOption matches a label exactly and the label carries duration and
  // format, so look the option's value up by its text instead.
  const reference = await page.locator("#sample-select option", { hasText: "ljspeech_reference" }).getAttribute("value");
  await page.locator("#sample-select").selectOption(reference);
  await page.locator("#clone-engine-select").selectOption("openvoice-v2");

  await page.locator("#script").fill("This is a cloned voice driving a talking avatar.");

  // The badge names the engine that actually spoke: no silent fallback.

  // Render the clone onto the face.
  await page.locator("#create-btn").click();
  await expect(page.locator("#speech-info")).toContainText("openvoice-v2", { timeout: 240_000 });
  await expect(page.getByText(/estimated, not measured/)).toHaveCount(0);
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 180_000 });

  // The video element decoded real frames, not just got a URL.
  const video = page.locator("#avatar-video");
  await expect.poll(() => video.evaluate((v) => v.readyState)).toBeGreaterThanOrEqual(1);
  const info = await video.evaluate((v) => ({ duration: v.duration, width: v.videoWidth, height: v.videoHeight }));
  expect(info.duration).toBeGreaterThan(1);
  expect(info.width).toBeGreaterThan(0);

  // And the file behind it is an MP4.
  const response = await page.request.get((await video.getAttribute("src")));
  expect(response.status()).toBe(200);
  const body = await response.body();
  expect(body.subarray(4, 8).toString()).toBe("ftyp");
});
