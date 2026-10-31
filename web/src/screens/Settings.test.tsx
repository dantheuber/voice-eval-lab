import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as Tooltip from "@radix-ui/react-tooltip";
import { ApiError } from "@/api/transport";
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

describe("switching profile", () => {
  beforeEach(() => { vi.mocked(endpoints.setActiveProfile).mockReset(); });
  const pick = async (user: ReturnType<typeof userEvent.setup>) => user.selectOptions(await screen.findByRole("combobox", { name: "Active profile" }), await screen.findByRole("option", { name: "local-fast" }));
  const notInstalled = new ApiError(400, { code: "not_installed", message: "Fast isn't installed. Missing .runtime/models/parakeet.gguf, and 1 more. Run scripts/setup.sh --profile local-fast.", detail: { profile: "local-fast", missing: [".runtime/models/parakeet.gguf", ".runtime/models/lfm3.gguf"], setup: "scripts/setup.sh --profile local-fast" } }, "x");

  it("names everything missing and the command that gets it, and it stays on screen", async () => {
    vi.mocked(endpoints.setActiveProfile).mockRejectedValue(notInstalled);
    const user = userEvent.setup();
    show();
    await pick(user);
    expect(await screen.findByText(".runtime/models/parakeet.gguf")).toBeInTheDocument();
    expect(screen.getByText(".runtime/models/lfm3.gguf")).toBeInTheDocument();
    expect(screen.getByText("scripts/setup.sh --profile local-fast")).toBeInTheDocument();
    expect(screen.getByText(/nothing changed/i)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Active profile" })).toHaveValue("local-default");
  });

  it("says what to fix when setup doesn't cover the profile", async () => {
    vi.mocked(endpoints.setActiveProfile).mockRejectedValue(new ApiError(400, { code: "not_installed", message: "x", detail: { profile: "mine", missing: ["/no/such/server"], setup: null } }, "x"));
    const user = userEvent.setup();
    show();
    await pick(user);
    expect(await screen.findByText(/fix the path/i)).toBeInTheDocument();
  });

  it("shows why a swap failed and that the old profile stayed", async () => {
    vi.mocked(endpoints.setActiveProfile).mockRejectedValue(new ApiError(503, { code: "profile_failed", message: "Fast didn't start, so Local stays active. llm: exited with code 1" }, "x"));
    const user = userEvent.setup();
    show();
    await pick(user);
    expect(await screen.findByText(/Local stays active/)).toBeInTheDocument();
  });

  it("holds the picker still while servers start, and clears the error on the next try", async () => {
    vi.mocked(endpoints.setActiveProfile).mockRejectedValueOnce(notInstalled);
    let finish = (_: Profile) => {};
    vi.mocked(endpoints.setActiveProfile).mockImplementationOnce(() => new Promise<Profile>((resolve) => { finish = resolve; }));
    const user = userEvent.setup();
    show();
    await pick(user);
    await screen.findByText(/nothing changed/i);
    await pick(user);
    expect(await screen.findByText(/starting local-fast/i)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Active profile" })).toBeDisabled();
    expect(screen.queryByText(/nothing changed/i)).not.toBeInTheDocument();
    finish(profile("local-fast"));
    await vi.waitFor(() => expect(screen.getByRole("combobox", { name: "Active profile" })).toBeEnabled());
  });
});

describe("model servers", () => {
  it("shows who holds a port in full, on as many lines as it takes", async () => {
    const detail = "port 18102 is held by llama-server (pid 4321), not by voicelab";
    show({ health: [{ ...healthy, state: "down", detail }] });
    expect(await screen.findByText(detail)).not.toHaveClass("nowrap");
  });
});

describe("AWS credentials", () => {
  it("says why boto3 couldn't use what it found", async () => {
    show({ aws: { region: null, credentials: "not_found", problem: "Partial credentials found in env, missing: AWS_SECRET_ACCESS_KEY", testedAgainstLive: false } });
    expect(await screen.findByText(/missing: AWS_SECRET_ACCESS_KEY/)).toBeInTheDocument();
  });
});
