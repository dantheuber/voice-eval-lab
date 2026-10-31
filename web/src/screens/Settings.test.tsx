import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as Tooltip from "@radix-ui/react-tooltip";
import { endpoints } from "@/api/client";
import type { AwsStatus, Health, Profile } from "@/api/types";
import { SettingsScreen } from "./Settings";

vi.mock("@/components/Shell", () => ({ Shell: ({ children }: { children: ReactNode }) => <div>{children}</div> }));
vi.mock("@/api/client", () => ({
  endpoints: {
    profiles: vi.fn(), roles: vi.fn(async () => ({ items: [] })), health: vi.fn(), aws: vi.fn(), setActiveProfile: vi.fn(),
    system: vi.fn(async () => ({ gpu: { device: "gpu", memoryUsedGb: 1, memoryTotalGb: 2 }, servers: [], urls: { local: "http://x", lan: "https://y" }, cert: "self-signed" })),
    thresholds: vi.fn(async () => ({ deadAirS: 1, bargeStopTargetS: 1, replyLatencyWarnS: 1, mosBaseline: 1, audioListeningMode: "advisory" })),
  },
}));

const profile = (id: string): Profile => ({ id, label: id, description: "", configPath: `config/profiles/${id}.yaml`, health: "ok", liveCapable: true, maxCalls: null, benchmark: null, untested: false });
const healthy: Health = { role: "llm_support", name: "support bot LLM", model: "qwen", state: "ok", detail: "ready", elapsedS: null, usualS: null };

function show({ aws = { region: null, credentials: "found", problem: null, testedAgainstLive: false } as AwsStatus, health = [healthy] } = {}) {
  vi.mocked(endpoints.profiles).mockResolvedValue({ items: [profile("local-default"), profile("local-fast")], activeId: "local-default" });
  vi.mocked(endpoints.health).mockResolvedValue({ items: health, ready: true });
  vi.mocked(endpoints.aws).mockResolvedValue(aws);
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Tooltip.Provider><SettingsScreen /></Tooltip.Provider></MemoryRouter></QueryClientProvider>);
}

describe("AWS credentials", () => {
  it("says why boto3 couldn't use what it found", async () => {
    show({ aws: { region: null, credentials: "not_found", problem: "Partial credentials found in env, missing: AWS_SECRET_ACCESS_KEY", testedAgainstLive: false } });
    expect(await screen.findByText(/missing: AWS_SECRET_ACCESS_KEY/)).toBeInTheDocument();
  });
});
