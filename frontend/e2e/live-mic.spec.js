// Microphone input to the live avatar (R-41), in the browser. Chromium's fake capture device plays
// a periodic beep, which is enough to see frames come back for streamed audio and the mouth move.
import { expect, test } from "@playwright/test";

test.use({
  permissions: ["microphone"],
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
});

test("speaking into the microphone streams audio and the avatar answers with frames", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page.locator("#tab-live").click(); // the live avatar is on its own tab
  await expect(page.locator("#live-avatar-select")).toHaveValue("demo");
  await page.locator("#live-start").click();
  await expect(page.locator("#live-panel")).toHaveAttribute("data-state", "ready", { timeout: 90_000 });

  await expect(page.locator("#live-mic")).toBeDisabled();            // no consent basis chosen yet
  await page.locator("#live-mic-consent").selectOption("speaker-recorded");
  await page.locator("#live-mic").click();
  await expect(page.locator("#live-mic")).toHaveText("Stop microphone");

  // About 3 s of microphone audio: 6 chunks of 0.5 s, 25 frames per second back.
  const panel = page.locator("#live-panel");
  await expect.poll(async () => Number(await panel.getAttribute("data-frames")), { timeout: 30_000 }).toBeGreaterThan(60);
  await page.locator("#live-mic").click();
  await expect(page.locator("#live-stats")).toContainText(/microphone: \d+ chunks \(audio-energy\)/, { timeout: 30_000 });
  await expect(page.locator("#live-error")).toHaveCount(0);
});
