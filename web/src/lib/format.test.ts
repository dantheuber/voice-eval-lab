import { describe, expect, it } from "vitest";
import { fmtClock, fmtDuration, interruptionRate, interruptionScale, sevClass } from "./format";

describe("format", () => {
  it("formats clock times", () => {
    expect(fmtClock(0)).toBe("00:00");
    expect(fmtClock(172)).toBe("02:52");
    expect(fmtClock(14.4, true)).toBe("00:14.4");
    expect(fmtClock(null)).toBe("--:--");
  });
  it("formats durations", () => {
    expect(fmtDuration(42)).toBe("42 s");
    expect(fmtDuration(408)).toBe("6 min 48 s");
    expect(fmtDuration(120)).toBe("2 min");
  });
  it("maps severity to UI buckets and advisory wins", () => {
    expect(sevClass("critical", false)).toBe("high");
    expect(sevClass("minor", false)).toBe("med");
    expect(sevClass("info", false)).toBe("low");
    expect(sevClass("critical", true)).toBe("adv");
  });
  it("round-trips the interruption scale", () => {
    expect(interruptionScale(0)).toBe(1);
    expect(interruptionScale(1)).toBe(5);
    expect(interruptionScale(0.4)).toBe(3);
    expect(interruptionRate(interruptionScale(0.75))).toBe(0.75);
  });
});

describe("usageLine", () => {
  it("counts calls and names the invalid ones", async () => {
    const { usageLine } = await import("./format");
    expect(usageLine({ calls: 16, counts: { invalid: 1 } })).toBe("16 calls, 1 invalid");
    expect(usageLine({ calls: 1, counts: { invalid: 0 } })).toBe("1 call");
  });
});

describe("splitSummary", () => {
  it("lifts the reason out of the verdict engine's summary and keeps the counts as detail", async () => {
    const { splitSummary } = await import("./format");
    const s = "Verdict: fail. Why: the bot did something forbidden: Renew for more than one year; the caller also missed a required checkpoint, which doesn't excuse this: Ask for the new expiry date. Criteria: 11 passed, 4 failed, 0 unknown. Findings: 8, of which 4 advisory, plus 3 detail(s). Not checked: audio_listening (none configured).";
    const r = splitSummary(s);
    expect(r.why).toBe("the bot did something forbidden: Renew for more than one year; the caller also missed a required checkpoint, which doesn't excuse this: Ask for the new expiry date.");
    expect(r.rest).toMatch(/^Criteria: 11 passed, 4 failed/);
    expect(r.rest).toContain("Not checked: audio_listening");
  });
  it("has no reason to lift when the summary is only counts", async () => {
    const { splitSummary } = await import("./format");
    expect(splitSummary("Verdict: pass. Criteria: 13 passed, 0 failed, 0 unknown.")).toEqual({ why: "", rest: "Criteria: 13 passed, 0 failed, 0 unknown." });
  });
  it("returns a plain summary whole and an empty one empty", async () => {
    const { splitSummary } = await import("./format");
    expect(splitSummary("Refund promised but not issued; wrong timeframe.")).toEqual({ why: "Refund promised but not issued; wrong timeframe.", rest: "" });
    expect(splitSummary("")).toEqual({ why: "", rest: "" });
    expect(splitSummary(null)).toEqual({ why: "", rest: "" });
  });
});
