import { describe, expect, it } from "vitest";
import { addCounts, ciText, intervalsOverlap, median, passPct, rateDirection, serialize, wilson } from "./stats";

describe("rateDirection", () => {
  it("judges on rates, so 2/2 -> 3/10 is worse", () => {
    expect(rateDirection({ pass: 2, n: 2 }, { pass: 3, n: 10 })).toBe("worse");
    expect(rateDirection({ pass: 1, n: 3 }, { pass: 2, n: 3 })).toBe("better");
    expect(rateDirection({ pass: 1, n: 2 }, { pass: 2, n: 4 })).toBe("same");
  });
  it("is null with no valid calls on either side", () => {
    expect(rateDirection({ pass: 0, n: 0 }, { pass: 1, n: 1 })).toBeNull();
  });
});

describe("intervalsOverlap", () => {
  it("small samples overlap, large disjoint ones do not", () => {
    expect(intervalsOverlap({ pass: 17, n: 23 }, { pass: 19, n: 24 })).toBe(true);
    expect(intervalsOverlap({ pass: 300, n: 500 }, { pass: 400, n: 500 })).toBe(false);
  });
  it("is null when a side is empty", () => {
    expect(intervalsOverlap({ pass: 0, n: 0 }, { pass: 1, n: 2 })).toBeNull();
  });
});

describe("serialize", () => {
  it("runs writes for one key in order and keeps going after a failure", async () => {
    const order: string[] = [];
    const slow = serialize("k", () => new Promise<string>((r) => setTimeout(() => { order.push("a"); r("a"); }, 20)));
    const failing = serialize("k", async () => { order.push("b"); throw new Error("no"); });
    const fast = serialize("k", async () => { order.push("c"); return "c"; });
    await expect(slow).resolves.toBe("a");
    await expect(failing).rejects.toThrow("no");
    await expect(fast).resolves.toBe("c");
    expect(order).toEqual(["a", "b", "c"]);
  });
});

describe("wilson", () => {
  it("returns null for n = 0", () => {
    expect(wilson(0, 0)).toBeNull();
  });
  it("matches a known value (17 of 23)", () => {
    const [lo, hi] = wilson(17, 23)!;
    expect(Math.round(lo * 100)).toBe(54);
    expect(Math.round(hi * 100)).toBe(87);
  });
  it("is wide for small n and narrow for large n", () => {
    const small = wilson(4, 5)!;
    const large = wilson(400, 500)!;
    expect(small[1] - small[0]).toBeGreaterThan(large[1] - large[0]);
  });
  it("stays inside 0..1 at the edges", () => {
    expect(wilson(0, 8)![0]).toBe(0);
    expect(wilson(8, 8)![1]).toBe(1);
  });
  it("formats as a percent range", () => {
    expect(ciText(17, 23)).toBe("54-87%");
    expect(ciText(0, 0)).toBe("");
  });
});

describe("counts", () => {
  it("excludes invalid from the pass rate denominator", () => {
    expect(passPct({ pass: 3, fail: 1, needsReview: 0, invalid: 4 })).toBe(75);
  });
  it("returns null with no valid calls", () => {
    expect(passPct({ pass: 0, fail: 0, needsReview: 0, invalid: 2 })).toBeNull();
  });
  it("adds counts", () => {
    expect(addCounts({ pass: 1, fail: 2, needsReview: 3, invalid: 4 }, { pass: 1, fail: 1, needsReview: 1, invalid: 1 })).toEqual({ pass: 2, fail: 3, needsReview: 4, invalid: 5 });
  });
  it("median handles even and odd", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(median([])).toBeNull();
  });
});
