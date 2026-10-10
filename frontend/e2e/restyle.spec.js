// Restyle form (T8.6): the styles come from the API, and a refused request shows the server's own
// reason before any Stable Diffusion work. The slow img2img run itself was verified live through
// the API (docs/12-PROGRESS.md), not on every E2E run.
import { expect, test } from "@playwright/test";

test("the restyle form offers the API's styles and shows its refusals", async ({ page }) => {
  await page.goto("/");
  await page.locator('[data-avatar="demo"]').click(); // pick it explicitly: the user's own faces may sort first
  await expect(page.locator("#faces")).toHaveAttribute("data-selected", "demo");

  await page.locator("#restyle-toggle").click();
  for (const style of ["realistic", "cartoon", "painting", "sketch"]) {
    await expect(page.locator("#restyle-style").getByRole("option", { name: style })).toHaveCount(1);
  }
  await expect(page.locator("#restyle-form")).toContainText("keeps the original's consent record");

  // "demo" is taken: the API answers 409 before any generation starts.
  await page.locator("#restyle-avatar-id").fill("demo");
  await page.locator("#restyle-btn").click();
  await expect(page.locator(".alert.error")).toContainText("already exists");
});
