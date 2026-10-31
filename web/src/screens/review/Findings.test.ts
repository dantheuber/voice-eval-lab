import { describe, expect, it } from "vitest";
import { dbActualRow, verdictAllowed } from "./Findings";
import { headlineNote } from "../Compare";

describe("dbActualRow", () => {
  it("only a pass claims success", () => {
    expect(dbActualRow("refunds", "pass", "")).toMatchObject({ k: "ok" });
    expect(dbActualRow("refunds", "fail", "No row added.")).toMatchObject({ k: "miss", s: expect.stringContaining("No row added.") });
    expect(dbActualRow("refunds", "unknown", "snapshot missing")).toMatchObject({ k: "unexp", s: expect.stringContaining("could not be checked") });
    expect(dbActualRow("refunds", null, "")).toMatchObject({ k: "exp", s: expect.stringContaining("not checked yet") });
    expect(dbActualRow("refunds", "unknown", "").s).not.toContain("as expected");
  });
});

describe("verdictAllowed", () => {
  it("blocks everything while evaluation is pending", () => {
    expect(verdictAllowed("pass", null, null, true)).toBe(false);
    expect(verdictAllowed("needs_review", "needs_review", null, true)).toBe(false);
  });
  it("locks pass and fail on an invalid run until it is explicitly un-marked", () => {
    expect(verdictAllowed("pass", "invalid", null, false)).toBe(false);
    expect(verdictAllowed("fail", "invalid", "invalid", false)).toBe(false);
    expect(verdictAllowed("needs_review", "invalid", null, false)).toBe(true);
    // choosing review is the clearing transition; pass and fail unlock afterwards
    expect(verdictAllowed("pass", "invalid", "needs_review", false)).toBe(true);
    expect(verdictAllowed("fail", "invalid", "needs_review", false)).toBe(true);
  });
});

describe("headlineNote", () => {
  it("reports overlap for small samples and no overlap for large disjoint ones", () => {
    expect(headlineNote({ pass: 17, n: 23 }, { pass: 19, n: 24 })).toMatchObject({ kind: "warn", text: expect.stringContaining("overlap") });
    const big = headlineNote({ pass: 300, n: 500 }, { pass: 400, n: 500 });
    expect(big.kind).toBe("info");
    expect(big.text).toContain("do not overlap");
    expect(big.text).toContain("20-point improvement");
    expect(big.text).not.toContain("15 points");
  });
  it("handles an empty side", () => {
    expect(headlineNote({ pass: 0, n: 0 }, { pass: 1, n: 2 }).kind).toBe("plain");
  });
});
