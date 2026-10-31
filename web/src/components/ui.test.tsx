import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import * as Tooltip from "@radix-ui/react-tooltip";
import { ErrorBox, ModeBadge, PassRate, RateBar, VerdictBadge } from "./ui";
import { ApiError } from "@/api/transport";

const wrap = (ui: React.ReactElement) => render(<Tooltip.Provider>{ui}</Tooltip.Provider>);

describe("badges", () => {
  it("verdicts carry a word, not only a colour", () => {
    wrap(<VerdictBadge v="needs_review" />);
    expect(screen.getByText("needs review")).toBeInTheDocument();
  });
  it("turn-based mode explains itself in the title", () => {
    wrap(<ModeBadge mode="turn_based" />);
    expect(screen.getByText("turn-based").closest(".mode")).toHaveAttribute("title", expect.stringContaining("not measured"));
  });
});

describe("rates", () => {
  it("rate bar has a full text label including invalid", () => {
    wrap(<RateBar counts={{ pass: 3, fail: 1, needsReview: 2, invalid: 4 }} />);
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", "3 pass, 1 fail, 2 needs review of 6 valid; 4 invalid excluded");
    expect(screen.getByText("+4 invalid")).toBeInTheDocument();
  });
  it("pass rate excludes invalid and shows an interval", () => {
    wrap(<PassRate counts={{ pass: 17, fail: 4, needsReview: 2, invalid: 1 }} />);
    expect(screen.getByText("74%")).toBeInTheDocument();
    expect(screen.getByText("54-87%, n=23")).toBeInTheDocument();
  });
});

describe("ErrorBox", () => {
  it("renders the model-not-ready state with the role and timing", () => {
    const err = new ApiError(503, { code: "model_not_ready", message: "loading", detail: { role: "judge LLM", elapsedS: 22, usualS: 40 } }, "x");
    wrap(<ErrorBox error={err} />);
    expect(screen.getByText(/judge LLM is loading/)).toBeInTheDocument();
    expect(screen.getByText(/usually under 40 s/)).toBeInTheDocument();
  });
  it("renders a plain error with a retry link", () => {
    wrap(<ErrorBox error={new Error("boom")} retry={() => undefined} what="calls" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load calls");
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });
});
