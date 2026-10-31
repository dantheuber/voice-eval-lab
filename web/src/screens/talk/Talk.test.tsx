import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import * as Tooltip from "@radix-ui/react-tooltip";
import { ApiError } from "@/api/transport";
import { endpoints } from "@/api/client";
import { TalkScreen } from "./Talk";

vi.mock("@/components/Shell", () => ({ Shell: ({ children }: { children: ReactNode }) => <div>{children}</div> }));
vi.mock("./useMic", () => ({ useMic: () => ({ level: 0, devices: [], outputs: [], error: null, stream: null, headphonesLikely: false }) }));
vi.mock("./useBotCall", () => ({ useBotCall: () => ({ state: { status: "idle", error: null, muted: false, lines: [], events: [] }, start: vi.fn(), end: vi.fn(), setMuted: vi.fn() }) }));
vi.mock("@/api/client", () => ({
  endpoints: {
    scenarios: vi.fn(async () => ({ items: [] })),
    personas: vi.fn(async () => ({ items: [] })),
    health: vi.fn(async () => ({ items: [], ready: true })),
    profiles: vi.fn(async () => ({ items: [], activeId: "local-realtime" })),
    createTalkSession: vi.fn(),
  },
}));

const busy = new ApiError(409, { code: "profile_busy", message: "Local takes 1 call at a time, and a bot call is on it. Wait for that call to end, about 40 s, then start again.", detail: { maxCalls: 1, waitS: 40 } }, "x");

function show() {
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><Tooltip.Provider><TalkScreen /></Tooltip.Provider></MemoryRouter></QueryClientProvider>);
}

describe("starting a call on a profile with no line free", () => {
  beforeEach(() => { vi.mocked(endpoints.createTalkSession).mockReset(); });

  it("says what to wait for and lets you try again", async () => {
    vi.mocked(endpoints.createTalkSession).mockRejectedValue(busy);
    const user = userEvent.setup();
    show();
    await user.click(await screen.findByRole("button", { name: /start call/i }));
    expect(await screen.findByText(/about 40 s/)).toBeInTheDocument();
    expect(screen.getByText(/line is busy/i)).toBeInTheDocument();
    await user.click(screen.getByRole("link", { name: "Try again" }));
    expect(endpoints.createTalkSession).toHaveBeenCalledTimes(2);
  });

  it("keeps other failures in the plain line beside the button", async () => {
    vi.mocked(endpoints.createTalkSession).mockRejectedValue(new ApiError(500, { code: "internal", message: "no webrtc" }, "x"));
    const user = userEvent.setup();
    show();
    await user.click(await screen.findByRole("button", { name: /start call/i }));
    expect(await screen.findByText("no webrtc")).toBeInTheDocument();
    expect(screen.queryByText(/line is busy/i)).not.toBeInTheDocument();
  });
});
