// Voice-to-avatar (T8.2), in the browser: a recording the user supplies (here, speech made on this
// page and downloaded, so no real person's voice is involved) is uploaded with no transcript; the
// server recognises the words, aligns them and renders. Real Whisper, aligner, renderer, VideoSeal.
import { expect, test } from "@playwright/test";

test("an uploaded recording with no transcript becomes a lip-synced video recorded as supplied", async ({ page }) => {
  test.setTimeout(420_000);
  await page.goto("/");
  await expect(page.locator("#avatar-select")).toHaveValue("demo");

  // A recording to upload: synthesise one, then fetch its bytes as a user would download it.
  await page.getByPlaceholder(/^Enter text/).fill("Good morning, this recording was made earlier and uploaded.");
  await page.getByRole("button", { name: "Generate Speech" }).click();
  await expect(page.locator("audio")).toHaveAttribute("src", /\/outputs\/speech\.wav\?t=\d+/, { timeout: 120_000 });
  const audioUrl = (await page.locator("audio").getAttribute("src")).split("?")[0];
  const recording = await (await page.request.get(audioUrl)).body();

  await page.locator("#own-audio-toggle").click();
  await expect(page.locator("#own-audio-btn")).toBeDisabled();          // no file, no consent basis yet
  await page.locator("#own-audio-file").setInputFiles({ name: "my-recording.wav", mimeType: "audio/wav", buffer: recording });
  await expect(page.locator("#own-audio-btn")).toBeDisabled();          // still no consent basis
  await page.locator("#own-audio-consent").selectOption("open-licence");
  await page.locator("#own-audio-btn").click();

  // The words were recognised (no transcript was given) and the render runs on the usual job view.
  await expect(page.locator("#own-audio-info")).toContainText(/Recognised \(en, [\d.]+ s\)/, { timeout: 180_000 });
  await expect(page.locator("#own-audio-info")).toContainText(/morning/i);
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 240_000 });
  await expect(page.locator("#provenance-line")).toContainText(/invisible watermark verified/);

  // The manifest says the audio was supplied, under the basis given, with ASR words.
  const manifest = await (await page.request.get(await page.locator("#manifest-link").getAttribute("href"))).json();
  const origin = manifest.inputs.audio.speechRecord.origin;
  expect(origin.type).toBe("supplied");
  expect(origin.consentBasis).toBe("open-licence");
  expect(origin.transcriptSource).toBe("asr");
});
