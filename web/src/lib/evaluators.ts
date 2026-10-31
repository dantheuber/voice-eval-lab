/* Evaluator and finding vocabulary. Lists come from the server's `evaluators` array; these maps only supply
   friendly labels and the expected set shown while an evaluation is still running. Unknown evaluators and
   categories fall back to their id with underscores turned into spaces, so new ones appear automatically. */
import type { EvaluatorRun, Finding } from "@/api/types";

export const evaluatorLabels: Record<string, string> = {
  transcript_judge: "transcript judge", caller_validity: "caller validity", state: "state assertions", timing: "timing",
  pronunciation: "pronunciation", audio_quality: "audio quality", audio_listening: "audio listening", delivery: "delivery (measured)",
  transcribe: "transcription", tool_attempts: "tool attempts",
};
export const evalName = (e: string) => evaluatorLabels[e] ?? e.replace(/_/g, " ");

/* Evaluators whose findings can flag for review but never fail a call. Delivery limits come from a small
   corpus; audio listening agrees with humans under 80% of the time. */
export const ADVISORY_EVALUATORS = new Set(["audio_listening", "delivery"]);
export function isAdvisoryEvaluator(name: string, findings: Finding[] = []): boolean {
  if (ADVISORY_EVALUATORS.has(name)) return true;
  const own = findings.filter((f) => f.evaluator === name);
  return own.length > 0 && own.every((f) => f.advisory);
}

/* The order the dashboard shows while results are still arriving. Server-provided evaluators keep their own order. */
export const EXPECTED_EVALUATORS = ["transcript_judge", "caller_validity", "state", "tool_attempts", "timing", "pronunciation", "delivery", "audio_quality", "audio_listening"];

export function orderedEvaluators(runs: EvaluatorRun[], findings: Finding[] = [], pending = false): string[] {
  const out: string[] = [];
  const add = (n: string) => { if (!out.includes(n)) out.push(n); };
  runs.forEach((r) => add(r.name));
  findings.forEach((f) => add(f.evaluator));
  if (pending) EXPECTED_EVALUATORS.forEach(add);
  return out.filter((n) => n !== "transcribe");
}

export const categoryLabels: Record<string, { label: string; note?: string }> = {
  critical_token_corroborated: { label: "critical token, two engines agree", note: "The bot's own STT and the independent ASR both heard the same wrong token, so this can fail the call." },
  critical_token_asr_only: { label: "critical token, heard by one engine", note: "Only the independent ASR heard this; the bot's STT did not confirm it. Advisory until a person checks the clip." },
  critical_token: { label: "critical token" },
  low_confidence_critical: { label: "low confidence on a critical token" },
  low_confidence: { label: "low ASR confidence" },
  word_mismatch: { label: "word mismatch" },
  unanswered_turn: { label: "unanswered turn", note: "The caller stopped and the bot said nothing before the caller spoke again or the call ended." },
  dead_air: { label: "dead air" }, barge_in_stop: { label: "slow stop after barge-in" }, overlap: { label: "overlap" }, slow_response: { label: "slow response" },
  rushed: { label: "speech too fast" }, dragging: { label: "speech too slow" }, halting: { label: "halting speech" }, chopped: { label: "chopped audio" },
  flat: { label: "flat pitch" }, noisy: { label: "noisy speech" }, abrupt_onset: { label: "abrupt start" }, abrupt_offset: { label: "abrupt stop" }, abrupt_cut: { label: "abrupt cut" },
  blocked_tool_call: { label: "blocked tool call", note: "The bot's argument guard stopped this call, so it never reached a tool and nothing changed. Advisory: it counts toward review and can't fail a call." },
  unguarded_tool_call: { label: "tool call the guard would have blocked", note: "The guard was off, so this call reached the tool. A made-up value here fails the call." },
  missing_change: { label: "expected change missing" }, unexpected_change: { label: "unexpected change" }, rule: { label: "rule" }, delivery: { label: "delivery impression", note: "A listening model's impression. Where the measured delivery row disagrees, the measurement wins." },
};
export function categoryLabel(f: Pick<Finding, "category">): string {
  return categoryLabels[f.category]?.label ?? f.category.replace(/_/g, " ");
}
export function categoryNote(f: Pick<Finding, "category">): string | undefined {
  return categoryLabels[f.category]?.note;
}

/* ---- delivery evaluator metrics (keys inside `metrics` stay snake_case; camelCase accepted too) ---- */
export interface UtteranceDelivery {
  speaker: string; utteranceId: string | null; startS: number; endS: number; words: number; speechS: number;
  wpm: number | null; pauseCount: number; longestPauseS: number; pauseRatio: number; chops: number; midWordChops: number;
  pitchMedianHz: number | null; pitchRangeSt: number | null; pitchIqrSt: number | null; snrDb: number | null;
  onsetMs: number | null; offsetMs: number | null; flags: string[];
}
export interface DeliveryMetrics {
  bot: UtteranceDelivery[]; caller: UtteranceDelivery[];
  medians: { wpm: number | null; pitchIqrSt: number | null; snrDb: number | null };
  flagCounts: Record<string, number>; limitation: string | null; wpmNote: string | null;
}
const pick = (o: Record<string, unknown>, snake: string, camel: string): unknown => (o[snake] !== undefined ? o[snake] : o[camel]);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
function utt(raw: Record<string, unknown>): UtteranceDelivery {
  const g = (s: string, c: string) => pick(raw, s, c);
  return {
    speaker: String(g("speaker", "speaker") ?? ""), utteranceId: (g("utterance_id", "utteranceId") as string | null) ?? null,
    startS: num(g("start_s", "startS")) ?? 0, endS: num(g("end_s", "endS")) ?? 0, words: num(g("words", "words")) ?? 0, speechS: num(g("speech_s", "speechS")) ?? 0,
    wpm: num(g("wpm", "wpm")), pauseCount: num(g("pause_count", "pauseCount")) ?? 0, longestPauseS: num(g("longest_pause_s", "longestPauseS")) ?? 0, pauseRatio: num(g("pause_ratio", "pauseRatio")) ?? 0,
    chops: num(g("chops", "chops")) ?? 0, midWordChops: num(g("mid_word_chops", "midWordChops")) ?? 0,
    pitchMedianHz: num(g("pitch_median_hz", "pitchMedianHz")), pitchRangeSt: num(g("pitch_range_st", "pitchRangeSt")), pitchIqrSt: num(g("pitch_iqr_st", "pitchIqrSt")), snrDb: num(g("snr_db", "snrDb")),
    onsetMs: num(g("onset_ms", "onsetMs")), offsetMs: num(g("offset_ms", "offsetMs")), flags: Array.isArray(g("flags", "flags")) ? (g("flags", "flags") as string[]) : [],
  };
}
export function parseDelivery(metrics: Record<string, unknown> | undefined | null): DeliveryMetrics | null {
  if (!metrics) return null;
  const bot = Array.isArray(metrics.bot) ? (metrics.bot as Record<string, unknown>[]).map(utt) : null;
  if (!bot) return null;
  const med = (pick(metrics, "medians", "medians") as Record<string, unknown> | undefined) ?? {};
  return {
    bot, caller: Array.isArray(metrics.caller) ? (metrics.caller as Record<string, unknown>[]).map(utt) : [],
    medians: { wpm: num(med.wpm), pitchIqrSt: num(pick(med, "pitch_iqr_st", "pitchIqrSt")), snrDb: num(pick(med, "snr_db", "snrDb")) },
    flagCounts: ((pick(metrics, "flag_counts", "flagCounts") as Record<string, number> | undefined) ?? {}),
    limitation: (metrics.limitation as string | undefined) ?? null, wpmNote: (pick(metrics, "wpm_note", "wpmNote") as string | undefined) ?? null,
  };
}
export function deliveryFor(runs: EvaluatorRun[] | undefined): DeliveryMetrics | null {
  const run = runs?.find((r) => r.name === "delivery" && r.status === "ok");
  return run ? parseDelivery(run.metrics) : null;
}
/** The bot utterance measurement that overlaps a time range, for checking an audio listener's claim. */
export function deliveryAt(d: DeliveryMetrics | null, startS: number | null, endS: number | null): UtteranceDelivery | null {
  if (!d || startS == null) return null;
  const e = endS ?? startS;
  return d.bot.find((u) => u.startS <= e && u.endS >= startS) ?? null;
}
export function deliveryLine(u: UtteranceDelivery): string {
  const parts = [u.wpm != null ? `${Math.round(u.wpm)} wpm` : "no rate", u.pitchIqrSt != null ? `pitch IQR ${u.pitchIqrSt.toFixed(1)} st` : "no pitch", u.snrDb != null ? `SNR ${Math.round(u.snrDb)} dB` : "no SNR", `${u.pauseCount} pause${u.pauseCount === 1 ? "" : "s"}`];
  return parts.join(", ");
}

/* One-line result per evaluator from its metrics. Unknown evaluators say "done". */
export function summarizeEvaluator(name: string, m: Record<string, unknown>): string {
  const g = (k: string) => m[k] as number | undefined;
  switch (name) {
    case "transcript_judge": return g("criteriaTotal") != null ? `${g("criteriaPass")} of ${g("criteriaTotal")} criteria pass${g("unknown") ? `, ${g("unknown")} uncertain` : ""}` : "done";
    case "caller_validity": return g("checkpointsTotal") != null ? `${g("checkpointsReached")} of ${g("checkpointsTotal")} checkpoints` : "done";
    case "state": return g("expected") != null ? (g("missing") ? `${g("missing")} of ${g("expected")} expected changes missing` : "all expected changes present") : "done";
    case "timing": return g("reply_p50_s") != null ? `p50 reply ${g("reply_p50_s")} s, ${g("overlaps") ?? 0} overlaps, ${g("dead_air_events") ?? 0} dead-air events` : "done";
    case "pronunciation": {
      const total = g("criticalTokens") ?? g("critical_total"), mism = g("mismatches") ?? g("critical_mismatches"), corr = g("critical_corroborated");
      if (total == null && mism == null) return "done";
      return `${mism ?? 0} critical token mismatch${mism === 1 ? "" : "es"}${total != null ? ` of ${total}` : ""}${corr != null ? `, ${corr} corroborated by both engines` : ""}`;
    }
    case "tool_attempts": return g("attempts") ? `${g("attempts")} tool call${g("attempts") === 1 ? "" : "s"} the guard objected to, ${g("mistakes") ?? 0} made up by the bot, ${g("stopped") ?? 0} stopped` : "no tool calls the guard objected to";
    case "audio_quality": return g("mos") != null ? `MOS ${g("mos")}${g("baseline") != null ? ` (baseline ${g("baseline")})` : ""}${m.clipping ? ", clipping" : ""}` : "done";
    case "audio_listening": return g("notes") != null ? `${g("notes")} advisory note${g("notes") === 1 ? "" : "s"}` : "done";
    case "delivery": {
      const d = parseDelivery(m); if (!d) return "done";
      const flags = Object.values(d.flagCounts).reduce((a, b) => a + b, 0);
      const med = [d.medians.wpm != null ? `median ${Math.round(d.medians.wpm)} wpm` : null, d.medians.snrDb != null ? `SNR ${Math.round(d.medians.snrDb)} dB` : null].filter(Boolean).join(", ");
      return `${d.bot.length} bot utterances measured${med ? `, ${med}` : ""}, ${flags} flag${flags === 1 ? "" : "s"}`;
    }
    default: return "done";
  }
}
