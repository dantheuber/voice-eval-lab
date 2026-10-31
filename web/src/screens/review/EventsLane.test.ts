import { describe, expect, it } from "vitest";
import { describeEvent, layoutEvents } from "./EventsLane";

const WIDTH = 1170;

describe("describeEvent", () => {
  it("marks a blocked attempt apart from a call that ran, and says why in the tooltip", () => {
    const error = "blocked before reaching the tool by argument_guard rule not_heard: pin not in anything the bot heard from the caller";
    const d = describeEvent({ type: "tool_call", id: "t", atS: 1, name: "verify_identity", arguments: { pin: "1234" }, result: { error: "x" }, error, durationMs: 1, guard: { rule: "not_heard", fields: ["pin"], stopped: true } });
    expect(d.kind).toBe("blocked");
    expect(d.label).toBe("blocked verify_identity");
    expect(d.title).toContain('"pin":"1234"');
    expect(d.title).toContain("argument_guard rule not_heard");
  });
});

describe("layoutEvents", () => {
  it("stacks labels that would cover each other", () => {
    const places = layoutEvents([{ atS: 20.7, label: "find_customer" }, { atS: 24.2, label: "barge-in" }], 88, WIDTH);
    expect(places.map((p) => p.row)).toEqual([0, 1]);
  });
  it("keeps labels on one row when there's room", () => {
    const places = layoutEvents([{ atS: 10, label: "find_customer" }, { atS: 60, label: "get_invoice" }], 88, WIDTH);
    expect(places.map((p) => p.row)).toEqual([0, 0]);
  });
  it("keeps a label at the very end inside the lane", () => {
    const [last] = layoutEvents([{ atS: 87.7, label: "issue_refund" }], 88, WIDTH);
    expect(last.x + "issue_refund".length * 6 + 30).toBeLessThanOrEqual(WIDTH);
  });
  it("never goes past three rows on a busy call", () => {
    const busy = Array.from({ length: 40 }, (_, i) => ({ atS: 20 + i * 0.2, label: "verify_identity" }));
    expect(Math.max(...layoutEvents(busy, 300, WIDTH).map((p) => p.row))).toBe(2);
  });
});
