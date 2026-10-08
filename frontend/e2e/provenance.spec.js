// Gate 5, in the browser: a rendered video shows that it is watermarked and carries a signed manifest,
// and the Verify panel tells apart the exact file, a file with no manifest, an edited manifest and a
// file that carries no marks at all. Real Kokoro, real renderer, real VideoSeal and AudioSeal.
import { expect, test } from "@playwright/test";

// One second of 24 kHz silence as a WAV: audio that certainly carries no watermark.
function silentWav() {
  const samples = 24000;
  const buffer = Buffer.alloc(44 + samples * 2);
  buffer.write("RIFF", 0); buffer.writeUInt32LE(36 + samples * 2, 4); buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16); buffer.writeUInt16LE(1, 20); buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(24000, 24); buffer.writeUInt32LE(48000, 28); buffer.writeUInt16LE(2, 32); buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36); buffer.writeUInt32LE(samples * 2, 40);
  return buffer;
}

test("a render is marked and signed, and the Verify panel judges files correctly", async ({ page }) => {
  test.setTimeout(420_000);
  await page.goto("/");
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");

  await page.locator("#script").fill("This video carries a hidden mark and a signed record.");
  await page.locator("#create-btn").click();
  await expect(page.locator("#render-status")).toHaveText("COMPLETED", { timeout: 300_000 });

  // The studio says the video is marked, with how many tag bits read back from the encoded file.
  await expect(page.locator("#provenance-line")).toContainText(/invisible watermark verified \(\d+\/128 bits\)/);
  const bits = Number((await page.locator("#provenance-line").textContent()).match(/\((\d+)\/128/)[1]);
  expect(bits).toBeGreaterThanOrEqual(96);
  const manifestUrl = await page.locator("#manifest-link").getAttribute("href");
  expect(manifestUrl).toMatch(/\.mp4\.manifest\.json$/);

  // Fetch the real video and its manifest the way a user would download them.
  const videoUrl = (await page.locator("#avatar-video").getAttribute("src")).split("?")[0];
  const video = await (await page.request.get(videoUrl)).body();
  const manifestText = await (await page.request.get(manifestUrl)).text();
  const manifest = JSON.parse(manifestText);
  expect(manifest.signature.algorithm).toBe("ed25519");
  expect(manifest.aiGenerated).toBe(true);

  await page.locator("#tab-verify").click(); // the Verify panel is on its own tab
  const verify = async (files) => {
    await page.locator("#verify-file").setInputFiles(files.file);
    if (files.manifest) await page.locator("#verify-manifest").setInputFiles(files.manifest);
    else await page.locator("#verify-manifest").setInputFiles([]);
    await page.locator("#verify-btn").click();
    await expect(page.locator("#verify-result")).toBeVisible({ timeout: 120_000 });
  };
  const verdict = () => page.locator("#verify-verdict").getAttribute("data-verdict");
  const mp4 = { name: "rendered.mp4", mimeType: "video/mp4", buffer: video };

  // 1. the exact file with its manifest
  await verify({ file: mp4, manifest: { name: "m.json", mimeType: "application/json", buffer: Buffer.from(manifestText) } });
  expect(await verdict()).toBe("authentic_original");
  await expect(page.locator("#verify-result")).toContainText("manifest: valid, ours, matches this file");

  // 2. the same video with no manifest: our watermark is still found
  await verify({ file: mp4 });
  expect(await verdict()).toBe("ours_modified");
  await expect(page.locator("#verify-result")).toContainText(/video watermark: found \(\d+\/128 bits match\)/);

  // 3. an edited manifest is called what it is, even though the watermark is present
  const edited = JSON.parse(manifestText);
  edited.inputs.avatar.consentBasis = "written-consent";
  await verify({ file: mp4, manifest: { name: "m.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(edited)) } });
  expect(await verdict()).toBe("tampered_manifest");
  await expect(page.locator("#verify-result")).toContainText("signature does NOT match");

  // 4. audio with no mark: no evidence, and the page says that is not proof of authenticity
  await verify({ file: { name: "silence.wav", mimeType: "audio/wav", buffer: silentWav() } });
  expect(await verdict()).toBe("no_evidence");
  await expect(page.locator("#verify-result")).toContainText("does NOT show the content is real");
});
