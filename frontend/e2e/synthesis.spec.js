// "Create video" with the standard voice, end to end: one click makes the speech (with measured phoneme
// timing) and the video. Real Kokoro, aligner and renderer.
import { expect, test } from "@playwright/test";

test("one click turns a script into speech and a finished video", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
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

test("the speaker follows the face and the chosen voice is what the server is asked for", async ({ page }) => {
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");
  // gen-live was generated as a woman: a female speaker is preselected and the label says why.
  await page.locator('[data-avatar="gen-live"]').click();
  await expect(page.locator("#speaker-select")).toHaveValue(/^af_/);
  await expect(page.getByText("(this face: female)")).toBeVisible();
  // demo has no recorded gender: the user is asked to pick, and the pick is sent with the request.
  await page.locator('[data-avatar="demo"]').click();
  await expect(page.locator("#speaker-hint")).toBeVisible();
  await page.locator("#speaker-select").selectOption("am_michael");
  await expect(page.locator("#speaker-hint")).toHaveCount(0);
  let sent = null;
  let calls = 0;
  await page.route("**/api/v1/audio/synthesize", async (route) => {
    calls += 1;
    sent = route.request().postDataJSON();
    await route.fulfill({ status: 400, json: { detail: "stopped by the test" } }); // no need to render here
  });
  await page.locator("#create-btn").dblclick(); // a double click must start one job, not two
  await expect(page.locator("#create-error")).toContainText("stopped by the test");
  await page.waitForTimeout(500);
  expect(calls).toBe(1);
  expect(sent.voice).toBe("am_michael");
  expect(sent.mode).toBe("fast");
});
