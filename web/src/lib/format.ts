import type { CallMode, Severity, Verdict } from "@/api/types";

export function fmtClock(s: number | null | undefined, tenths = false): string {
  if (s == null || !Number.isFinite(s)) return "--:--";
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  if (tenths) return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(Math.floor(r)).padStart(2, "0")}`;
}

export function fmtDuration(s: number): string {
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60), r = Math.round(s - m * 60);
  return r ? `${m} min ${r} s` : `${m} min`;
}

export function fmtWhen(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function fmtAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "";
  const diff = Math.max(0, now - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return `${Math.round(diff / 86400)} d ago`;
}

export const verdictLabel: Record<Verdict | "running" | "queued" | "evaluating" | "failed", string> = {
  pass: "pass", fail: "fail", needs_review: "needs review", invalid: "invalid",
  running: "running", queued: "queued", evaluating: "evaluating", failed: "failed",
};

export const modeLabel: Record<CallMode, string> = { live_bots: "live", turn_based: "turn-based", human: "mic", upload: "upload" };
export const modeTitle: Record<CallMode, string> = {
  live_bots: "Live, real time. Timing metrics are real.",
  turn_based: "Turn-based, slower than real time. Gaps between turns are synthetic, so timing metrics are not measured.",
  human: "Human caller through the browser microphone. Live.",
  upload: "Uploaded recording. Reduced evidence.",
};

/** UI severity buckets: the API's four levels plus advisory. */
export type SevClass = "high" | "med" | "low" | "adv";
export function sevClass(severity: Severity, advisory: boolean): SevClass {
  if (advisory) return "adv";
  if (severity === "critical" || severity === "major") return "high";
  if (severity === "minor") return "med";
  return "low";
}
export const sevLabel: Record<SevClass, string> = { high: "high", med: "med", low: "low", adv: "advisory, cannot fail the call" };

export { evalName, evaluatorLabels as evaluatorLabel } from "./evaluators";

export function interruptionScale(rate: number): number {
  return Math.min(5, Math.max(1, Math.round(rate * 4) + 1));
}
export function interruptionRate(scale: number): number {
  return (Math.min(5, Math.max(1, scale)) - 1) / 4;
}

/** The verdict engine's summary reads "Verdict: fail. Why: <reason> Criteria: ... Findings: ... Not checked: ...".
    The reason is what a reviewer needs first; the rest is detail for a disclosure. Plain summaries come back whole. */
export function splitSummary(summary: string | null | undefined): { why: string; rest: string } {
  const text = (summary ?? "").trim();
  if (!text) return { why: "", rest: "" };
  const body = text.replace(/^Verdict:\s*[a-z_ ]+\.\s*/i, "");
  if (/^(Criteria|Findings|Not checked|Rules|Beats):\s/.test(body)) return { why: "", rest: body };
  const m = /^Why:\s*/i.exec(body);
  const afterWhy = m ? body.slice(m[0].length) : body;
  const cut = afterWhy.search(/\s(Criteria|Findings|Not checked|Rules|Beats):\s/);
  if (!m && cut < 0) return { why: afterWhy, rest: "" };
  if (cut < 0) return { why: afterWhy.trim(), rest: "" };
  return { why: afterWhy.slice(0, cut).trim(), rest: afterWhy.slice(cut + 1).trim() };
}

/** "16 calls, 1 invalid": how many calls a scenario or persona has been used in, and how many of those do not count. */
export function usageLine(u: { calls: number; counts: { invalid: number } }): string {
  const calls = `${u.calls} call${u.calls === 1 ? "" : "s"}`;
  return u.counts.invalid ? `${calls}, ${u.counts.invalid} invalid` : calls;
}

export function shortId(id: string): string {
  return id.length > 14 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}
