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
