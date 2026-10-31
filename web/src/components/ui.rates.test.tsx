import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PassFrac, PassRate, RateBar } from "./ui";

describe("RateBar numeric honesty", () => {
  it("normalises segments over valid calls only and keeps invalid outside the bar", () => {
    const { container } = render(<RateBar counts={{ pass: 3, fail: 1, needsReview: 0, invalid: 4 }} />);
    const segs = [...container.querySelectorAll(".rate i")] as HTMLElement[];
    const flex = segs.map((s) => Number(s.style.flexGrow || s.style.flex));
    const total = flex.reduce((a, b) => a + b, 0);
    expect(total).toBe(4); // valid only
    expect(flex[0] / total).toBeCloseTo(0.75); // pass segment matches the 75% pass rate shown beside it
    expect(container.querySelector(".rate .i")).toBeNull();
    expect(screen.getByText("+4 invalid")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", "3 pass, 1 fail, 0 needs review of 4 valid; 4 invalid excluded");
  });
  it("shows an empty bar when there are no valid calls", () => {
    const { container } = render(<RateBar counts={{ pass: 0, fail: 0, needsReview: 0, invalid: 2 }} />);
    expect(container.querySelector(".rate .none")).not.toBeNull();
    expect(container.querySelector(".rate .p")).toBeNull();
  });
});

describe("PassRate display", () => {
  it("shows percentage, interval, and valid n together", () => {
    render(<PassRate counts={{ pass: 17, fail: 4, needsReview: 2, invalid: 1 }} />);
    expect(screen.getByText("74%")).toBeInTheDocument();
    expect(screen.getByText("54-87%, n=23")).toBeInTheDocument();
  });
  it("compact form keeps n inline and the interval in the title", () => {
    render(<PassFrac pass={3} n={6} compact />);
    expect(screen.getByText("50%")).toBeInTheDocument();
    expect(screen.getByText("n6")).toBeInTheDocument();
    expect(screen.getByTitle(/95% interval 19-81%/)).toBeInTheDocument();
  });
  it("never divides by zero", () => {
    render(<PassFrac pass={0} n={0} />);
    expect(screen.getByText("no valid calls")).toBeInTheDocument();
  });
});
