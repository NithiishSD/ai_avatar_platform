// Gate 1, face half: the studio draws the face mesh onto the photo, in the
// right place. Measured from the canvas pixels, not judged by eye.
import { expect, test } from "@playwright/test";

const MESH = [0x6e, 0xe7, 0xb7]; // #6ee7b7, face landmarks
const IRIS = [0xf4, 0x72, 0xb6]; // #f472b6, the 10 iris points

// Runs in the page: counts pixels near a colour and returns their centroid.
// Antialiased dots blend into the photo, hence the tolerance.
function measure([canvasSel, colours]) {
  const canvas = document.querySelector(canvasSel);
  const { width, height } = canvas;
  const data = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  return colours.map(([r, g, b]) => {
    let n = 0, sx = 0, sy = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (Math.abs(data[i] - r) + Math.abs(data[i + 1] - g) + Math.abs(data[i + 2] - b) < 40) {
        const p = i / 4;
        n += 1; sx += p % width; sy += Math.floor(p / width);
      }
    }
    return { n, cx: n ? sx / n : 0, cy: n ? sy / n : 0, width, height };
  });
}

test("the face mesh is drawn onto the photo, inside the face", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#avatar-select")).toHaveValue("demo");

  // Pose and blendshapes come from the live analysis of this photo.
  await expect(
    page.getByText(/478 landmarks · yaw -?[\d.]+° · pitch -?[\d.]+° · roll -?[\d.]+° · 52 blendshapes/),
  ).toBeVisible();

  const canvas = "#avatar-landmark-canvas";
  // Drawing happens after the image loads; poll until the mesh is there.
  await expect.poll(async () => (await page.evaluate(measure, [canvas, [MESH]]))[0].n).toBeGreaterThan(500);
  const [mesh, iris] = await page.evaluate(measure, [canvas, [MESH, IRIS]]);
  expect(mesh.width).toBe(512); // drawn at the photo's own resolution
  expect(iris.n).toBeGreaterThan(0);

  // The mesh sits on the face: its centroid is inside the face box that the
  // API reports for this photo.
  const form = { avatarId: "demo" };
  const analysis = (await (await page.request.post("http://localhost:8000/api/v1/avatar/face/analyze", { multipart: form })).json()).analysis;
  const box = analysis.boundingBox;
  expect(mesh.cx).toBeGreaterThan(box.x);
  expect(mesh.cx).toBeLessThan(box.x + box.width);
  expect(mesh.cy).toBeGreaterThan(box.y);
  expect(mesh.cy).toBeLessThan(box.y + box.height);

  // Turning the overlay off leaves just the photo.
  await page.getByLabel("show landmarks").uncheck();
  await expect.poll(async () => (await page.evaluate(measure, [canvas, [MESH]]))[0].n).toBeLessThan(mesh.n / 20);
});
