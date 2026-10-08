// Voice-to-avatar (T8.2) in the browser: a recording the user supplies (here, speech made by the API
// beforehand, so no real person's voice is involved) is uploaded with no transcript; the server
// recognises the words, aligns them and renders. Real Whisper, aligner, renderer, VideoSeal.
import { expect, test } from "@playwright/test";

const API = "http://localhost:8000";

test("an uploaded recording with no transcript becomes a lip-synced video recorded as supplied", async ({ page, request }) => {
  test.setTimeout(420_000);
  // A recording to upload, made through the API and downloaded as a user would.
  const made = await (await request.post(`${API}/api/v1/audio/synthesize`, {
    data: { text: "Good morning, this recording was made earlier and uploaded.", mode: "fast", outputFilename: "e2e-recording.wav" },
  })).json();
  expect(made.status).toBe("SUCCESS");
  const recording = await (await request.get(`${API}/outputs/e2e-recording.wav`)).body();

  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");
  await page.locator("#voice-own").click();
  await expect(page.locator("#create-btn")).toBeDisabled();          // no file, no consent basis yet
  await page.locator("#own-audio-file").setInputFiles({ name: "my-recording.wav", mimeType: "audio/wav", buffer: recording });
  await expect(page.locator("#create-btn")).toBeDisabled();          // still no consent basis
  await page.locator("#own-audio-consent").selectOption("open-licence");
  await page.locator("#create-btn").click();

  // The words were recognised (no transcript was given) and the render runs on the usual result view.
  await expect(page.locator("#speech-info")).toContainText(/Words recognised \(en\)/, { timeout: 180_000 });
  await expect(page.locator("#speech-info")).toContainText(/morning/i);
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 240_000 });
  await expect(page.locator("#provenance-line")).toContainText(/invisible watermark verified/);

  // The manifest says the audio was supplied, under the basis given, with recognised words.
  const manifest = await (await page.request.get(await page.locator("#manifest-link").getAttribute("href"))).json();
  const origin = manifest.inputs.audio.speechRecord.origin;
  expect(origin.type).toBe("supplied");
  expect(origin.consentBasis).toBe("open-licence");
  expect(origin.transcriptSource).toBe("asr");
});
