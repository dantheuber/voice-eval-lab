import { describe, expect, it } from "vitest";
import { categoryLabel, categoryNote, deliveryAt, evalName, isAdvisoryEvaluator, orderedEvaluators, parseDelivery, summarizeEvaluator } from "./evaluators";
import type { EvaluatorRun, Finding } from "@/api/types";

const run = (name: string, metrics: Record<string, unknown> = {}): EvaluatorRun => ({ name, status: "ok", reason: "", durationS: 1, model: null, metrics });
const finding = (evaluator: string, category: string, advisory: boolean): Finding => ({ id: `${evaluator}-${category}`, evaluator, category, severity: "minor", advisory, title: "", detail: "", evidence: [], clipStartS: 1, clipEndS: 2, criterionId: null, humanVerdict: null, humanNote: "", humanAt: null });

describe("orderedEvaluators", () => {
  it("shows server evaluators in server order and includes new ones", () => {
    expect(orderedEvaluators([run("delivery"), run("timing"), run("brand_new")])).toEqual(["delivery", "timing", "brand_new"]);
  });
  it("adds the expected set only while pending, and drops transcribe", () => {
    expect(orderedEvaluators([run("transcribe"), run("timing")], [], true)).toEqual(["timing", "transcript_judge", "caller_validity", "state", "tool_attempts", "pronunciation", "delivery", "audio_quality", "audio_listening"]);
    expect(orderedEvaluators([run("timing")], [], false)).toEqual(["timing"]);
  });
  it("includes evaluators known only from findings", () => {
    expect(orderedEvaluators([], [finding("delivery", "rushed", true)])).toEqual(["delivery"]);
  });
});

describe("advisory and category labels", () => {
  it("knows the advisory evaluators and infers from findings", () => {
    expect(isAdvisoryEvaluator("delivery")).toBe(true);
    expect(isAdvisoryEvaluator("pronunciation", [finding("pronunciation", "critical_token_asr_only", true)])).toBe(true);
    expect(isAdvisoryEvaluator("pronunciation", [finding("pronunciation", "critical_token_corroborated", false)])).toBe(false);
  });
  it("labels the new pronunciation and timing categories in plain words", () => {
    expect(categoryLabel({ category: "critical_token_corroborated" })).toBe("critical token, two engines agree");
    expect(categoryLabel({ category: "critical_token_asr_only" })).toBe("critical token, heard by one engine");
    expect(categoryLabel({ category: "unanswered_turn" })).toBe("unanswered turn");
    expect(categoryLabel({ category: "something_else" })).toBe("something else");
  });
  it("names blocked attempts and says what they do to a verdict", () => {
    expect(categoryLabel({ category: "blocked_tool_call" })).toBe("blocked tool call");
    expect(categoryNote({ category: "blocked_tool_call" })).toContain("never reached a tool");
    expect(categoryLabel({ category: "unguarded_tool_call" })).toBe("tool call the guard would have blocked");
    expect(evalName("tool_attempts")).toBe("tool attempts");
  });
  it("sums up blocked attempts", () => {
    expect(summarizeEvaluator("tool_attempts", { attempts: 3, mistakes: 2, stopped: 3 })).toBe("3 tool calls the guard objected to, 2 made up by the bot, 3 stopped");
    expect(summarizeEvaluator("tool_attempts", { attempts: 0, mistakes: 0, stopped: 0 })).toBe("no tool calls the guard objected to");
  });
});

describe("delivery metrics", () => {
  const metrics = {
    bot: [{ speaker: "bot", utterance_id: "u1", start_s: 10, end_s: 14, words: 12, speech_s: 3.5, wpm: 205.7, pause_count: 1, longest_pause_s: 0.4, pause_ratio: 0.1, chops: 0, mid_word_chops: 0, voiced_s: 2, pitch_median_hz: 180, pitch_range_st: 6.2, pitch_std_st: 1.1, pitch_iqr_st: 1.9, snr_db: 27.4, onset_ms: 120, offset_ms: 90, flags: ["rushed"] }],
    caller: [], medians: { wpm: 160, pitch_iqr_st: 2.4, snr_db: 26 }, flag_counts: { rushed: 1 }, limitation: "small corpus",
  };
  it("parses snake_case keys from the evaluator", () => {
    const d = parseDelivery(metrics)!;
    expect(d.bot[0].wpm).toBeCloseTo(205.7);
    expect(d.bot[0].pitchIqrSt).toBeCloseTo(1.9);
    expect(d.medians.snrDb).toBe(26);
    expect(d.flagCounts.rushed).toBe(1);
  });
  it("finds the utterance under a time range", () => {
    const d = parseDelivery(metrics);
    expect(deliveryAt(d, 11, 12)?.utteranceId).toBe("u1");
    expect(deliveryAt(d, 30, 31)).toBeNull();
  });
  it("summarises delivery and pronunciation corroboration", () => {
    expect(summarizeEvaluator("delivery", metrics)).toBe("1 bot utterances measured, median 160 wpm, SNR 26 dB, 1 flag");
    expect(summarizeEvaluator("pronunciation", { criticalTokens: 14, mismatches: 2, critical_corroborated: 1 })).toBe("2 critical token mismatches of 14, 1 corroborated by both engines");
    expect(summarizeEvaluator("brand_new", {})).toBe("done");
  });
});
