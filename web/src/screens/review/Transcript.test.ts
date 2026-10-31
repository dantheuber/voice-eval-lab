import { describe, expect, it } from "vitest";
import { diffWords, findingsByUtterance, sameWords } from "./Transcript";
import { reviewA } from "@/api/mock/fixtures";

describe("diffWords", () => {
  it("highlights a substituted token", () => {
    const d = diffWords("I can see invoice FH-4419 was charged", "I can see invoice FH forty-four ninety was charged");
    expect(d.pre).toBe("I can see invoice");
    expect(d.del).toBe("FH-4419");
    expect(d.ins).toBe("FH forty-four ninety");
    expect(d.post).toBe("was charged");
  });
  it("handles identical strings", () => {
    const d = diffWords("same words", "same words");
    expect(d.del).toBe("");
    expect(d.ins).toBe("");
  });
});

describe("what real ASR changes that isn't a word", () => {
  it("treats punctuation, case, and hyphens as the same words", () => {
    expect(sameWords("Thanks for calling Fernhill Internet, this is Fern.", "Thanks for calling Fernhill Internet. This is Fern.")).toBe(true);
    expect(sameWords("forty nine dollars", "forty-nine dollars")).toBe(true);
    expect(sameWords("forty nine dollars", "fifty nine dollars")).toBe(false);
  });
  it("doesn't widen a real substitution to the punctuation around it", () => {
    const d = diffWords("One moment, let me check FH-4419 now.", "One moment. Let me check FH-4490 now.");
    expect(d.del).toBe("FH-4419");
    expect(d.ins).toBe("FH-4490");
  });
});

describe("findingsByUtterance", () => {
  it("pins the pronunciation finding on the bot turn it overlaps", () => {
    const m = findingsByUtterance(reviewA);
    const botTurn = reviewA.call.timeline.find((e) => e.type === "utterance" && e.speaker === "bot" && e.startS === 45.3)!;
    expect(m.get(botTurn.id)?.map((f) => f.id)).toContain("fnd_a2");
  });
  it("gives every finding with a clip a home", () => {
    const m = findingsByUtterance(reviewA);
    const placed = new Set([...m.values()].flat().map((f) => f.id));
    for (const f of reviewA.eval!.findings) expect(placed.has(f.id)).toBe(true);
  });
});

describe("a finding that covers several moments", () => {
  const utterance = (id: string, startS: number, endS: number) => ({ ...reviewA.call.timeline.find((e) => e.type === "utterance")!, id, startS, endS });
  const rollup = { ...reviewA.eval!.findings[0], id: "f_roll", evaluator: "delivery", clipStartS: 7, clipEndS: 13, evidence: [{ quote: "", speaker: "bot", startS: 10, endS: 12 }, { quote: "", speaker: "bot", startS: 100, endS: 102 }] };
  const payload = (findings: typeof reviewA.eval.findings) => ({ ...reviewA, call: { ...reviewA.call, timeline: [utterance("near", 9, 13), utterance("far", 99, 103), utterance("clean", 50, 53)] }, eval: { ...reviewA.eval!, findings } }) as typeof reviewA;

  it("attaches to every utterance its evidence touches", () => {
    const by = findingsByUtterance(payload([rollup]));
    expect(by.get("near")?.map((f) => f.id)).toEqual(["f_roll"]);
    expect(by.get("far")?.map((f) => f.id)).toEqual(["f_roll"]);
    expect(by.has("clean")).toBe(false);
  });
  it("lists a finding once on an utterance two of its moments share", () => {
    const close = { ...rollup, evidence: [{ quote: "", speaker: "bot", startS: 10, endS: 10.5 }, { quote: "", speaker: "bot", startS: 11, endS: 11.5 }] };
    expect(findingsByUtterance(payload([close] as never)).get("near")).toHaveLength(1);
  });
  it("still finds a home for evidence in a gap between utterances", () => {
    const gap = { ...rollup, clipStartS: null, clipEndS: null, evidence: [{ quote: "", speaker: null, startS: 14, endS: 20 }] };
    expect(findingsByUtterance(payload([gap] as never)).get("near")).toHaveLength(1);
  });
});
