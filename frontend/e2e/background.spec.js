// Render option: the photo's background is replaced, and the pixels of the
// decoded video prove it (not just the API's say-so).
import { expect, test } from "@playwright/test";


// Draws the video's frame near t=0.1 s onto a canvas and returns it. The seek
// and the retry exist because a canvas drawn before the browser has painted a
// decoded frame reads as pure black, which looks like a wrong video but is not.
async function readFrame(video) {
  return video.evaluate(async (v) => {
    v.currentTime = 0.1;
    if (v.seeking) await new Promise((resolve) => v.addEventListener("seeked", resolve, { once: true }));
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    const at = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data.slice(0, 3));
    for (let attempt = 0; attempt < 10; attempt += 1) {
      ctx.drawImage(v, 0, 0);
      const corners = [at(6, 6), at(c.width - 7, 6)];
      if (corners.some(([r, g, b]) => r + g + b > 0)) {
        return { duration: v.duration, width: v.videoWidth, corners };
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    return { duration: v.duration, width: v.videoWidth, corners: [at(6, 6), at(c.width - 7, 6)] };
  });
}

test("a chosen background colour is what the rendered video shows", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");

  await page.locator("#script").fill("A new background behind the speaker.");

  await page.locator("#background-toggle").check();
  await page.locator("#background-color").fill("#00c800");
  await page.locator("#create-btn").click();
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 120_000 });
  await expect(page.getByText("background: color #00c800")).toBeVisible();

  // Draw the video's first decoded frame to a canvas and read its corners.
  const video = page.locator("#avatar-video");
  await expect.poll(() => video.evaluate((v) => v.readyState)).toBeGreaterThanOrEqual(2);
  const { corners } = await readFrame(video);
  for (const [r, g, b] of corners) {
    expect(Math.abs(r - 0) + Math.abs(g - 200) + Math.abs(b - 0)).toBeLessThan(60); // H.264 is lossy
  }
});
