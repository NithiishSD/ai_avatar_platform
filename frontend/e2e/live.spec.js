// Gate 4, in the browser: a live session streams speech and animated frames
// into the page while the text is still being synthesised. Real Kokoro, real
// aligner, real animator, over a real WebSocket.
import { expect, test } from "@playwright/test";

// Starts a sampler that fingerprints the lower half of the live canvas every 40 ms.
// Distinct fingerprints = the picture really changed (the mouth moved), not just "something drew".
async function startCanvasSampler(page) {
  await page.evaluate(() => {
    window.__fingerprints = new Set();
    window.__sampler = setInterval(() => {
      const canvas = document.querySelector("#live-canvas");
      const { data } = canvas.getContext("2d").getImageData(0, canvas.height >> 1, canvas.width, canvas.height >> 1);
      let h = 0;
      for (let i = 0; i < data.length; i += 13) h = (h * 31 + data[i]) | 0;
      window.__fingerprints.add(h);
    }, 40);
  });
}

async function openSession(page) {
  await page.goto("/");
  await page.locator("#tab-live").click(); // the live avatar is on its own tab
  await expect(page.locator("#live-avatar-select")).toHaveValue("demo");
  await page.locator("#live-start").click();
  await expect(page.locator("#live-panel")).toHaveAttribute("data-state", "ready", { timeout: 90_000 });
}

test("a live session streams audio chunks and a moving picture into the page", async ({ page }) => {
  test.setTimeout(240_000);
  await openSession(page);
  await expect(page.locator("#live-status")).toContainText("384×384 @ 25 fps · kokoro");
  await startCanvasSampler(page);

  await page.locator("#live-text").fill("This is the first live sentence. And this is the second one.");
  await page.locator("#live-speak").click();
  await expect(page.locator("#live-stats")).toContainText("finished: 2 sentences", { timeout: 120_000 });

  const panel = page.locator("#live-panel");
  expect(Number(await panel.getAttribute("data-chunks"))).toBe(2);        // one audio chunk per sentence
  expect(Number(await panel.getAttribute("data-frames"))).toBeGreaterThan(60); // ~2-4 s at 25 fps
  await expect(page.locator("#live-stats")).toContainText(/first audio \d+ ms · first frame \d+ ms/);

  // The picture is a face that changed while it spoke, not a blank or frozen canvas.
  const fingerprints = await page.evaluate(() => window.__fingerprints.size);
  expect(fingerprints).toBeGreaterThan(3);
  const lit = await page.evaluate(() => {
    const c = document.querySelector("#live-canvas");
    const { data } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    let bright = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 60) bright += 1;
    return bright / (data.length / 4);
  });
  expect(lit).toBeGreaterThan(0.3); // a portrait fills the frame; an undrawn canvas would be 0

  // A second utterance on the same session works (the timeline keeps running).
  await page.locator("#live-text").fill("One more thing.");
  await page.locator("#live-speak").click();
  await expect(page.locator("#live-stats")).toContainText("finished: 1 sentences", { timeout: 60_000 });
});

test("interrupt stops the speech and the session stays usable; stop ends it", async ({ page }) => {
  test.setTimeout(240_000);
  await openSession(page);
  const eight = Array.from({ length: 8 }, (_, i) => `This is sentence number ${i + 1} of a long speech.`).join(" ");
  await page.locator("#live-text").fill(eight);
  await page.locator("#live-speak").click();
  const panel = page.locator("#live-panel");
  await expect.poll(async () => Number(await panel.getAttribute("data-frames")), { timeout: 60_000 }).toBeGreaterThan(5);

  // Audio for the coming sentences is already queued in the browser: Interrupt must stop it too,
  // not just the picture (the owner heard the voice carry on, M-05).
  await expect.poll(async () => Number(await panel.getAttribute("data-playing")), { timeout: 30_000 }).toBeGreaterThan(0);
  await page.locator("#live-interrupt").click();
  await expect(panel).toHaveAttribute("data-playing", "0", { timeout: 1_000 });
  await expect(panel).toHaveAttribute("data-state", "ready", { timeout: 20_000 });
  await page.waitForTimeout(500); // anything already on the wire lands
  const settled = Number(await panel.getAttribute("data-frames"));
  await page.waitForTimeout(2500);
  expect(Number(await panel.getAttribute("data-frames"))).toBe(settled); // nothing more arrives
  expect(settled).toBeLessThan(8 * 50); // the speech was cut short: eight sentences are ~8 x 60 frames

  await page.locator("#live-text").fill("Back again.");
  await page.locator("#live-speak").click();
  await expect(page.locator("#live-stats")).toContainText("finished: 1 sentences", { timeout: 60_000 });

  await page.locator("#live-stop").click();
  await expect(panel).toHaveAttribute("data-state", "closed", { timeout: 10_000 });
});
