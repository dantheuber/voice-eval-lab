import { describe, expect, it } from "vitest";
import { clipRange, dbActualRow, dbGlance, evaluatorsGlance, momentClip, moments, rubricGlance, verdictAllowed } from "./Findings";
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

describe("moments", () => {
  const finding = (evidence: { quote: string; speaker: "bot" | "caller" | null; startS: number | null; endS: number | null }[]) => ({ id: "f", clipStartS: 7, clipEndS: 13, evidence }) as never;

  it("lists every timed occurrence, quoted or not", () => {
    const f = finding([{ quote: "", speaker: "bot", startS: 100, endS: 102 }, { quote: "hello", speaker: "bot", startS: 10, endS: 12 }, { quote: "", speaker: null, startS: null, endS: null }]);
    expect(moments(f).map((m) => m.startS)).toEqual([10, 100]);
  });
  it("needs more than one to be worth listing", () => {
    expect(moments(finding([{ quote: "", speaker: "bot", startS: 10, endS: 12 }]))).toEqual([]);
  });
  it("gives a start with no end a playable range", () => {
    const f = finding([{ quote: "", speaker: "bot", startS: 10, endS: null }, { quote: "", speaker: "bot", startS: 20, endS: 21 }]);
    expect(moments(f)[0]).toMatchObject({ startS: 10, endS: 10 });
  });
  it("plays its own moment, not the representative clip", () => {
    const f = finding([{ quote: "", speaker: "bot", startS: 10, endS: 12 }, { quote: "", speaker: "bot", startS: 100, endS: 102 }]);
    expect(clipRange(momentClip(f, moments(f)[1]), 300)).toEqual([99, 103]);
    expect(clipRange(f, 300)).toEqual([6, 14]);
  });
});

describe("section glances", () => {
  it("summarise the rubric and flag anything that is not a pass", () => {
    const c = (status: "pass" | "fail" | "unknown") => ({ criterionId: "x", kind: "must", text: "", status, evidence: [], reason: "" }) as never;
    expect(rubricGlance([c("pass"), c("pass"), c("fail"), c("unknown")])).toEqual({ text: "2 of 4 pass, 1 fail, 1 undecided", attention: true });
    expect(rubricGlance([c("pass")])).toEqual({ text: "1 of 1 pass", attention: false });
    expect(rubricGlance([])).toEqual({ text: "", attention: false });
  });
  it("summarise database changes, missing first", () => {
    expect(dbGlance([{ k: "exp", s: "" }, { k: "miss", s: "" }, { k: "unexp", s: "" }, { k: "unexp", s: "" }], true)).toEqual({ text: "1 missing, 2 unexpected", attention: true });
    expect(dbGlance([{ k: "exp", s: "" }, { k: "ok", s: "" }], true)).toEqual({ text: "as expected", attention: false });
    expect(dbGlance([], false)).toEqual({ text: "not available", attention: false });
  });
  it("count evaluator runs and only flag errors", () => {
    const e = (status: "ok" | "skipped" | "error") => ({ name: "x", status, reason: "", durationS: 0, model: null, metrics: {} });
    expect(evaluatorsGlance([e("ok"), e("ok"), e("skipped")], false)).toEqual({ text: "2 ran, 1 skipped", attention: false });
    expect(evaluatorsGlance([e("ok"), e("error")], true)).toEqual({ text: "1 ran, 1 failed, still running", attention: true });
  });
});
