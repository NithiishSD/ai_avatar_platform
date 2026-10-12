// N-15 / R-44 in the browser: the look sliders change the running live session's picture in real
// time. Real server, real face, real WebSocket: a slider move must change the pixels on the canvas
// and the server must acknowledge it, a reset must bring the picture back, and speech must still
// work with a look applied.
import { expect, test } from "@playwright/test";

// Mean lightness and mean red-minus-blue of the live canvas, read from its pixels.
async function canvasStats(page) {
  return page.evaluate(() => {
    const c = document.querySelector("#live-canvas");
    const { data } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    let luma = 0;
    let rb = 0;
    const n = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      luma += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      rb += data[i] - data[i + 2];
    }
    return { luma: luma / n, rb: rb / n };
  });
}

async function setSlider(page, id, value) {
  // A range input is set through the native setter and an input event, which is what a drag fires.
  await page.evaluate(({ id, value }) => {
    const el = document.getElementById(id);
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(el, String(value));
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, { id, value });
}

test("the look sliders change the live picture in real time and reset restores it", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await page.locator("#tab-live").click();
  await page.locator("#live-avatar-select").selectOption("demo");
  await page.locator("#live-start").click();
  await expect(page.locator("#live-panel")).toHaveAttribute("data-state", "ready", { timeout: 90_000 });

  // The picture is there before anything is said (the preview frame sent after `ready`).
  await expect.poll(async () => (await canvasStats(page)).luma, { timeout: 20_000 }).toBeGreaterThan(20);
  const before = await canvasStats(page);

  await page.locator("#live-look > summary").click();
  await expect(page.locator("#live-look-controls")).toBeVisible({ timeout: 20_000 });
  // 25 expression + 15 look rows are built from the catalogue (autoLighting is a checkbox)
  await expect(page.locator("#live-look-controls input[type=range]")).toHaveCount(39);

  // Brightness up: the server acknowledges and the canvas gets lighter.
  const applied0 = Number(await page.locator("#live-panel").getAttribute("data-look-applied"));
  await setSlider(page, "live-look-appearance-brightness", 0.6);
  await expect(page.locator("#live-panel")).toHaveAttribute("data-look-applied", String(applied0 + 1), { timeout: 20_000 });
  await expect.poll(async () => (await canvasStats(page)).luma, { timeout: 20_000 }).toBeGreaterThan(before.luma + 8);

  // Warmth up as well: red rises against blue.
  await setSlider(page, "live-look-appearance-warmth", 0.8);
  await expect.poll(async () => (await canvasStats(page)).rb, { timeout: 20_000 }).toBeGreaterThan(before.rb + 8);
  await expect(page.locator("#live-look-controls")).toHaveAttribute("data-changed", "2");

  // Speech still works with the look applied, and the first frame is lit like the preview.
  await page.locator("#live-text").fill("The look stays while I speak.");
  await page.locator("#live-speak").click();
  await expect(page.locator("#live-stats")).toContainText("finished: 1 sentences", { timeout: 120_000 });
  expect((await canvasStats(page)).luma).toBeGreaterThan(before.luma + 5);

  // Reset puts the original picture back.
  await page.locator("#live-look-reset").click();
  await expect.poll(async () => (await canvasStats(page)).luma, { timeout: 20_000 }).toBeLessThan(before.luma + 4);
  await expect(page.locator("#live-look-controls")).toHaveAttribute("data-changed", "0");
});

test("a standing expression and head position move the face on the live canvas", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page.locator("#tab-live").click();
  await page.locator("#live-avatar-select").selectOption("demo");
  await page.locator("#live-start").click();
  await expect(page.locator("#live-panel")).toHaveAttribute("data-state", "ready", { timeout: 90_000 });
  await expect.poll(async () => (await canvasStats(page)).luma, { timeout: 20_000 }).toBeGreaterThan(20);

  // A fingerprint of the whole canvas; a moved face changes it.
  const fingerprint = () => page.evaluate(() => {
    const c = document.querySelector("#live-canvas");
    const { data } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
    let h = 0;
    for (let i = 0; i < data.length; i += 7) h = (h * 31 + data[i]) | 0;
    return h;
  });
  const rest = await fingerprint();
  await page.locator("#live-look > summary").click();
  await expect(page.locator("#live-look-controls")).toBeVisible({ timeout: 20_000 });

  await page.locator("#live-look-expression > summary").click();
  await setSlider(page, "live-look-appearance-expression-jawOpen", 0.7);
  await expect.poll(fingerprint, { timeout: 20_000 }).not.toBe(rest);
  const opened = await fingerprint();

  await page.locator("#live-look-head > summary").click();
  await setSlider(page, "live-look-appearance-headRoll", 8);
  await expect.poll(fingerprint, { timeout: 20_000 }).not.toBe(opened);
});

test("a look chosen on the create screen is rendered into the video and reported with it", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click();
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");
  await page.locator("#script").fill("A look chosen before the render.");
  await page.locator("#create-look > summary").click();
  await expect(page.locator("#create-look-controls")).toBeVisible({ timeout: 20_000 });
  await setSlider(page, "create-look-appearance-saturation", -1);
  await setSlider(page, "create-look-appearance-vignette", 0.8);
  await expect(page.locator("#create-look-controls")).toHaveAttribute("data-changed", "2");
  await page.locator("#create-btn").click();
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 180_000 });
  // The result line names the look that was applied, from the server's own report.
  await expect(page.getByText(/look: saturation, vignette/)).toBeVisible();
  // And the picture really is desaturated: take a frame and compare its colour spread.
  const spread = await page.evaluate(async () => {
    const video = document.querySelector("#avatar-video");
    video.currentTime = 0.5;
    await new Promise((r) => { video.onseeked = r; setTimeout(r, 3000); });
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    const g = c.getContext("2d");
    g.drawImage(video, 0, 0);
    const { data } = g.getImageData(0, 0, c.width, c.height);
    let s = 0;
    for (let i = 0; i < data.length; i += 4) s += Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]);
    return s / (data.length / 4);
  });
  expect(spread).toBeLessThan(6); // the unmodified demo face averages about 25
});

