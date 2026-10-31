import { defineConfig } from "@playwright/test";

// Runs against the Vite dev server in fixture mode, so no backend is needed.
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:5199",
    headless: true,
    viewport: { width: 1440, height: 900 },
    launchOptions: { executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome" },
  },
  webServer: {
    command: "VITE_API_MOCK=1 pnpm exec vite --port 5199 --strictPort",
    url: "http://127.0.0.1:5199",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
