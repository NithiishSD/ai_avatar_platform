// "Create video" with the standard voice, end to end: one click makes the speech (with measured phoneme
// timing) and the video. Real Kokoro, aligner and renderer.
import { expect, test } from "@playwright/test";

test("one click turns a script into speech and a finished video", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");
  await page.locator("#script").fill("Hello from the end to end test.");
  await page.locator("#create-btn").click();
  await expect(page.locator("#create-btn")).toHaveText(/Making the voice|Animating the face/);
  await expect(page.locator("#speech-info")).toContainText("Voice: kokoro", { timeout: 120_000 });
  await expect(page.getByText(/estimated, not measured/)).toHaveCount(0);
  const audio = page.locator("#speech-audio");
  await expect(audio).toHaveAttribute("src", /\/outputs\/studio-\d+\.wav$/);
  const response = await page.request.get(await audio.getAttribute("src"));
  const body = await response.body();
  expect(body.subarray(0, 4).toString()).toBe("RIFF");
  expect(body.length).toBeGreaterThan(24_000); // more than half a second of 24 kHz audio
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 180_000 });
  await expect(page.locator("#avatar-video")).toBeVisible();
  await expect(page.locator("#create-btn")).toHaveText("Create video");
});
