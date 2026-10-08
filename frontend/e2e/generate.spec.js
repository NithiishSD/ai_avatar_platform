// Synthetic face form: choices come from the API, and a refused request
// shows the server's own reason. The slow Stable Diffusion run itself is
// verified live by hand (docs/12-PROGRESS.md), not on every E2E run.
import { expect, test } from "@playwright/test";

test("the generate form offers the API's fixed choices and shows its refusals", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");

  await page.locator("#generate-face-toggle").click();
  // Options are served by GET /avatar/generate/options, not hardcoded.
  await expect(page.getByRole("option", { name: "bald" })).toHaveCount(1);
  await expect(page.getByRole("option", { name: "older" })).toHaveCount(1);

  // "demo" is taken: the API answers 409 before any generation starts.
  await page.locator("#generate-avatar-id").fill("demo");
  await page.locator("#generate-face-btn").click();
  await expect(page.locator(".alert.error")).toContainText("already exists");
});
