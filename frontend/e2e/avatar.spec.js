// Face picker (step 1 of "Create video"): the registered faces come from the API as portrait cards,
// the first usable one is chosen, and clicking another card chooses it.
import { expect, test } from "@playwright/test";

test("the face picker shows the registered faces and a click chooses one", async ({ page, request }) => {
  const faces = (await (await request.get("http://localhost:8000/api/v1/avatar/faces")).json()).avatars;
  await page.goto("/");
  const picker = page.locator("#faces");
  // The first usable face in the API's order is chosen (the user's own faces may come before demo).
  await expect(picker).toHaveAttribute("data-selected", faces.find((f) => f.usable).avatarId);
  await expect(picker.locator(".face")).toHaveCount(faces.length);
  // Each card shows the face's own image, served by the API.
  const src = await picker.locator('[data-avatar="demo"] img').getAttribute("src");
  expect((await request.get(src)).headers()["content-type"]).toMatch(/^image\//);
  const other = faces.find((f) => f.usable && f.avatarId !== faces.find((g) => g.usable).avatarId);
  test.skip(!other, "only one usable face registered");
  await picker.locator(`[data-avatar="${other.avatarId}"]`).click();
  await expect(picker).toHaveAttribute("data-selected", other.avatarId);
  await expect(picker.locator(`[data-avatar="${other.avatarId}"]`)).toHaveAttribute("aria-pressed", "true");
});
