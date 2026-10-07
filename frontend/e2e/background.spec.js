// Render option: the photo's background is replaced, and the pixels of the
// decoded video prove it (not just the API's say-so).
import { expect, test } from "@playwright/test";

test("a chosen background colour is what the rendered video shows", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await expect(page.locator("#avatar-select")).toHaveValue("demo");

  await page.getByPlaceholder(/^Enter text/).fill("A new background behind the speaker.");
  await page.getByRole("button", { name: "Generate Speech" }).click();
  await expect(page.locator("audio")).toHaveAttribute("src", /\/outputs\/speech\.wav\?t=\d+/, { timeout: 120_000 });

  await page.locator("#background-toggle").check();
  await page.locator("#background-color").fill("#00c800");
  await page.locator("#render-video-btn").click();
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 120_000 });
  await expect(page.getByText("background: color #00c800")).toBeVisible();

  // Draw the video's first decoded frame to a canvas and read its corners.
  const video = page.locator("#avatar-video");
  await expect.poll(() => video.evaluate((v) => v.readyState)).toBeGreaterThanOrEqual(2);
  const corners = await video.evaluate((v) => {
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0);
    const at = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data.slice(0, 3));
    return [at(6, 6), at(c.width - 7, 6)]; // top corners; the label is bottom-left
  });
  for (const [r, g, b] of corners) {
    expect(Math.abs(r - 0) + Math.abs(g - 200) + Math.abs(b - 0)).toBeLessThan(60); // H.264 is lossy
  }
});
