// Gate 3, in the browser: a custom (generated) avatar, a cloned voice speaking
// another language with an emotion, and a replaced background, rendered into
// one preview video. Real Stable Diffusion (first run only), real XTTS-v2,
// real alignment, real renderer.
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

const API = "http://localhost:8000";
const FACE = "gate3-face";

// Generates the custom avatar through the API the first time and reuses it
// afterwards (it lives in inputs/faces, which is gitignored like the others).
async function ensureCustomAvatar(request) {
  const faces = await (await request.get(`${API}/api/v1/avatar/faces`)).json();
  if (faces.avatars.some((a) => a.avatarId === FACE && a.usable)) return;
  const start = await request.post(`${API}/api/v1/avatar/generate`, {
    data: { avatarId: FACE, age: "middle-aged", presentation: "woman", hair: "short-dark", seed: 11, attempts: 4, steps: 20 },
  });
  expect(start.status(), await start.text()).toBe(202);
  const { taskId } = await start.json();
  await expect
    .poll(async () => (await (await request.get(`${API}/api/v1/avatar/generate/${taskId}`)).json()).status, {
      timeout: 480_000, intervals: [3000],
    })
    .toBe("COMPLETED");
}

test("custom avatar + cloned Spanish voice + emotion + new background -> preview video", async ({ page, request }) => {
  test.setTimeout(900_000);
  await ensureCustomAvatar(request);

  await page.goto("/");
  await page.locator(`[data-avatar="${FACE}"]`).click();
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", FACE);

  // Cloned voice, speaking Spanish (the reference is English).
  await page.locator("#voice-clone").click();
  const reference = await page.locator("#sample-select option", { hasText: "ljspeech_reference" }).getAttribute("value");
  await page.locator("#sample-select").selectOption(reference);
  await page.locator("#clone-engine-select").selectOption("xtts-v2");
  await page.locator("#language-search").fill("spanish");
  await expect(page.locator("#language-select option[value='spa']")).toHaveCount(1);
  await page.locator("#language-select").selectOption("spa");

  // An emotion preset.
  await page.locator('[data-emotion="joy"]').click();

  await page.locator("#script").fill("Hola, esta es mi voz clonada hablando español con alegría.");

  // The badge names the engine that really spoke; the emotion card says what it did.

  // New background, then render.
  await page.locator("#background-toggle").check();
  await page.locator("#background-color").fill("#0b3d91");
  await page.locator("#create-btn").click();
  await expect(page.locator("#speech-info")).toContainText("xtts-v2", { timeout: 300_000 });
  await expect(page.getByText(/estimated, not measured/)).toHaveCount(0);
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 300_000 });
  await expect(page.getByText("background: color #0b3d91")).toBeVisible();

  // The decoded video: right size, and its top corners are the new colour.
  const video = page.locator("#avatar-video");
  await expect.poll(() => video.evaluate((v) => v.readyState)).toBeGreaterThanOrEqual(2);
  const probe = await readFrame(video);
  expect(probe.duration).toBeGreaterThan(2);
  expect(probe.width).toBeLessThanOrEqual(512); // PREVIEW tier
  for (const [r, g, b] of probe.corners) {
    expect(Math.abs(r - 11) + Math.abs(g - 61) + Math.abs(b - 145)).toBeLessThan(70);
  }
});
