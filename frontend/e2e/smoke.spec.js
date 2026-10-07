// The studio loads and talks to a live backend. These run against real
// servers started by playwright.config.js.
import { expect, test } from "@playwright/test";

test("studio loads and reports a healthy backend", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AI Avatar Creator Studio" })).toBeVisible();
  // The health check runs on mount and prints "<status> (<queueBackend>)".
  await expect(page.getByText(/^ok \(in_memory\)$/)).toBeVisible();
});

test("the language catalogue loads from the backend", async ({ page }) => {
  await page.goto("/");
  const select = page.locator("#language-select");
  // English is offered statically; anything beyond it came from the API.
  await expect.poll(() => select.locator("option").count()).toBeGreaterThan(1);
});

test("switching to clone mode loads the voice references", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Synthesis Mode").selectOption("clone");
  const samples = page.locator("#sample-select option");
  await expect.poll(() => samples.count()).toBeGreaterThan(0);
  await expect(page.locator("#sample-select")).toContainText("ljspeech_reference");
});

test("a stale language lookup cannot overwrite a newer one", async ({ page }) => {
  await page.goto("/");
  const select = page.locator("#language-select");
  await expect.poll(() => select.locator("option").count()).toBeGreaterThan(2);
  const values = await select.locator("option").evaluateAll((opts) => opts.map((o) => o.value));
  const [, slow, fast] = values;

  // Hold the first language's lookup for 2.5 s so its reply arrives *after*
  // the second one. Before the ignore-flag fix, that late reply won.
  await page.route(`**/api/v1/audio/languages/${slow}`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    await route.continue();
  });

  await select.selectOption(slow);
  await select.selectOption(fast);
  const info = page.getByText(new RegExp(`🔤 .*\\(${fast}\\)`));
  await expect(info).toBeVisible();
  // Wait past the delayed reply, then check the newer answer still stands.
  await page.waitForTimeout(3500);
  await expect(info).toBeVisible();
  await expect(page.getByText(new RegExp(`🔤 .*\\(${slow}\\)`))).toHaveCount(0);
});

test("the avatar panel shows landmarks for the demo face", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#avatar-select")).toHaveValue("demo");
  // 478 is MediaPipe's FaceLandmarker count; the line also carries the pose.
  await expect(page.getByText(/478 landmarks · yaw/)).toBeVisible();
  await expect(page.locator("#avatar-landmark-canvas")).toBeVisible();
});
