import { expect, test } from "@playwright/test";

/* The review keyboard loop, in fixture mode: land on the first open finding, play its clip, decide, advance,
   set the call verdict, move to the next call. */
test.describe("call review keyboard loop", () => {
  test("decides findings and moves to the next call without the mouse", async ({ page }) => {
    await page.goto("/calls/call_a3f2c1d0e9b7?theme=dark");
    await expect(page.getByText("finding 1 of 6")).toBeVisible();
    // first open finding is selected and its clip is cued
    const first = page.locator("[data-finding='fnd_a1']");
    await expect(first).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText(/clip 00:14 to 00:23/)).toBeVisible();

    // Enter plays the clip through a real <audio> element (fixture audio is a synthesised WAV blob)
    await page.keyboard.press("Enter");
    await expect(page.getByRole("button", { name: "Pause", exact: true })).toBeVisible({ timeout: 8000 });
    await expect.poll(async () => page.locator("text=/^\\d\\d:\\d\\d\\.\\d/").first().textContent(), { timeout: 5000 }).not.toMatch(/^00:14\.0/);
    await page.keyboard.press(" ");
    await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();

    // y confirms and advances to finding 2
    await page.keyboard.press("y");
    await expect(first.getByText("confirmed")).toBeVisible();
    await expect(page.getByText("finding 2 of 6")).toBeVisible();
    await expect(page.getByText("4 open", { exact: true })).toBeVisible();

    // n dismisses finding 2 and advances
    await page.keyboard.press("n");
    await expect(page.locator("[data-finding='fnd_a2']").getByText("dismissed")).toBeVisible();
    await expect(page.getByText("finding 3 of 6")).toBeVisible();

    // c opens a note on the selected finding, Ctrl+Enter saves it
    await page.keyboard.press("c");
    const ta = page.locator("textarea");
    await expect(ta).toBeFocused();
    await ta.fill("Bot kept going after barge-in, confirmed by ear.");
    await page.keyboard.press("Control+Enter");
    await expect(page.getByText("Bot kept going after barge-in, confirmed by ear.")).toBeVisible();

    // j / k move the selection
    await page.keyboard.press("j");
    await expect(page.getByText("finding 4 of 6")).toBeVisible();
    await page.keyboard.press("k");
    await expect(page.getByText("finding 3 of 6")).toBeVisible();

    // 2 sets the human verdict to fail
    await page.keyboard.press("2");
    await expect(page.getByRole("group", { name: "Set human verdict" }).getByRole("button", { name: /fail/ })).toHaveAttribute("aria-pressed", "true");

    // ] goes to the next call needing review (oldest first, never back to this one)
    await page.keyboard.press("]");
    await expect(page).not.toHaveURL(/call_a3f2c1d0e9b7/);
    await expect(page).toHaveURL(/\/calls\/call_/);
    await expect(page.getByText(/finding \d+ of \d+|no findings|evaluating/)).toBeVisible();
  });

  test("turn-based calls say what was not measured", async ({ page }) => {
    await page.goto("/calls/call_1c3d0a1b2c3d?theme=dark");
    await expect(page.getByText("Turn-based call.")).toBeVisible();
    await expect(page.getByText("not measured (turn-based)")).toBeVisible();
    await expect(page.getByText("gaps synthetic")).toBeVisible();
    await expect(page.getByText("advisory, cannot fail the call").first()).toBeVisible();
    await expect(page.getByText("critical token, heard by one engine")).toBeVisible();
  });

  test("decisions persist across navigation in fixture mode", async ({ page }) => {
    await page.goto("/calls/call_a3f2c1d0e9b7?theme=dark");
    await expect(page.getByText("finding 1 of 6")).toBeVisible();
    await page.keyboard.press("y");
    await expect(page.locator("[data-finding='fnd_a1']").getByText("confirmed")).toBeVisible();
    await page.waitForTimeout(800); // let the (mock-latency) save land before a hard navigation
    await page.goto("/calls?theme=dark");
    await expect(page.getByRole("heading", { name: "Calls" })).toBeVisible();
    await page.goto("/calls/call_a3f2c1d0e9b7?theme=dark");
    await expect(page.locator("[data-finding='fnd_a1']").getByText("confirmed")).toBeVisible();
  });

  test("overview and calls list render with filters in the URL", async ({ page }) => {
    await page.goto("/?theme=dark");
    await expect(page.getByText("Needs a human").first()).toBeVisible();
    await page.goto("/calls?verdict=fail&theme=dark");
    const rows = page.locator("table.t tbody tr");
    await expect(rows).toHaveCount(3);
    await page.keyboard.press("j");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/calls\/call_/);
  });
});
