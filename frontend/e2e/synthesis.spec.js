// Gate 1, speech half: text typed into the studio becomes real speech from
// a real model, with phoneme timing measured from that audio.
import { expect, test } from "@playwright/test";

test("text becomes speech with measured phoneme timing", async ({ page }) => {
  await page.goto("/");
  // By placeholder, not label: the textarea sits inside its <label>, and the
  // accessible-name rules fold an embedded text box's current value into the
  // label's name, so "Text" alone never matches exactly.
  await page.getByPlaceholder(/^Enter text/).fill("Hello from the end to end test.");
  await page.getByRole("button", { name: "Generate Speech" }).click();

  // The player only gets a source once the synthesis task reports SUCCESS.
  // Kokoro loads on first use, which is slow on a CPU, hence the long wait.
  const audio = page.locator("audio");
  await expect(audio).toHaveAttribute("src", /\/outputs\/speech\.wav\?t=\d+/, { timeout: 120_000 });

  // The badge names the engine that actually spoke (no silent fallback).
  await expect(page.locator(".info-card", { hasText: "Model Used" })).toContainText("kokoro");

  // Forced alignment ran and produced phonemes...
  const phonemes = page.getByText(/^Aligned Phonemes \(\d+\)$/);
  await expect(phonemes).toBeVisible();
  const count = Number((await phonemes.textContent()).match(/\d+/)[0]);
  expect(count).toBeGreaterThan(5);
  // ...measured from the audio, not estimated from the text.
  await expect(page.getByText(/estimated, not measured/)).toHaveCount(0);

  // And the file behind the player is real audio, not an empty response.
  const response = await page.request.get(await audio.getAttribute("src"));
  expect(response.status()).toBe(200);
  const body = await response.body();
  expect(body.subarray(0, 4).toString()).toBe("RIFF");
  expect(body.length).toBeGreaterThan(24_000); // more than half a second of 24 kHz audio
});
