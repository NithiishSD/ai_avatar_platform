// The studio's shell and the controls that load data from the backend.
import { expect, test } from "@playwright/test";

test("studio loads and reports a healthy backend", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "AI Avatar Creator Studio" })).toBeVisible();
  await expect(page.locator("#backend-status")).toHaveText(/^ok \(in_memory\)$/);
  for (const tab of ["Create video", "Live avatar", "Verify a file"]) {
    await expect(page.getByRole("tab", { name: tab })).toBeVisible();
  }
});

test("the language catalogue loads from the backend", async ({ page }) => {
  await page.goto("/");
  const select = page.locator("#language-select");
  await expect.poll(() => select.locator("option").count()).toBeGreaterThan(1);
  await expect(select).toHaveValue("en");
});

test("choosing 'Clone a voice' loads the voice references", async ({ page }) => {
  await page.goto("/");
  await page.locator("#voice-clone").click();
  const samples = page.locator("#sample-select option");
  await expect.poll(() => samples.count()).toBeGreaterThan(0);
  await expect(page.locator("#sample-select")).toContainText("ljspeech_reference");
});

test("a slow, stale language search cannot overwrite a newer one", async ({ page }) => {
  await page.goto("/");
  const select = page.locator("#language-select");
  // The search for "tamil" is held back; "swahili" is typed after it and answers first.
  await page.route("**/api/v1/audio/languages?q=tamil*", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    await route.continue();
  });
  await page.locator("#language-search").fill("tamil");
  await page.waitForTimeout(400); // past the 250 ms debounce, so the tamil request is in flight
  await page.locator("#language-search").fill("swahili");
  await expect(select.locator("option", { hasText: "Swahili" }).first()).toBeAttached();
  await page.waitForTimeout(3000); // the held tamil reply arrives now, and must be ignored
  await expect(select.locator("option", { hasText: "Swahili" }).first()).toBeAttached();
  await expect(select.locator("option", { hasText: "Tamil" })).toHaveCount(0);
});

test("the two other tabs open the live avatar and the verify panel", async ({ page }) => {
  await page.goto("/");
  await page.locator("#tab-live").click();
  await expect(page.locator("#live-start")).toBeVisible();
  await page.locator("#tab-verify").click();
  await expect(page.locator("#verify-btn")).toBeVisible();
  await expect(page.locator("#create-btn")).toBeHidden();
});
