import { useEffect, useRef, useState } from "react";
import type { Finding, ReviewPayload, Verdict } from "@/api/types";
import { Button, HealthDot, Icon, Kbd, VerdictBadge } from "@/components/ui";
import { evalName, fmtClock, sevClass, sevLabel, verdictLabel } from "@/lib/format";
import { useReview } from "@/store/review";
import s from "./review.module.css";
import { pinLabel } from "./Transcript";

export const clipRange = (f: Finding, durationS: number): [number, number] => [Math.max(0, (f.clipStartS ?? 0) - 1), Math.min(durationS, (f.clipEndS ?? (f.clipStartS ?? 0) + 5) + 1)];

export function FindingCard({ payload, f, onPlay, onDecide, onSaveNote }: {
  payload: ReviewPayload; f: Finding; onPlay: (f: Finding) => void; onDecide: (f: Finding, v: "confirmed" | "dismissed") => void; onSaveNote: (f: Finding, note: string) => void;
}) {
  const selected = useReview((st) => st.selected) === f.id;
  const noteOpen = useReview((st) => st.noteOpen) === f.id;
  const setNoteOpen = useReview((st) => st.setNoteOpen);
  const select = useReview((st) => st.select);
  const sc = sevClass(f.severity, f.advisory);
  const [draft, setDraft] = useState(f.humanNote);
  const ta = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (noteOpen) { setDraft(f.humanNote); ta.current?.focus(); } }, [noteOpen, f.humanNote]);
  const [a, b] = clipRange(f, payload.call.durationS);
  const intendedHeard = f.evaluator === "pronunciation" && f.detail.includes("Intended:");
  const save = () => { onSaveNote(f, draft.trim()); setNoteOpen(null); };
  return (
    <div className={`${s.finding} ${selected ? s.sel : ""} ${f.advisory ? s.advisory : ""}`} data-finding={f.id} role="button" tabIndex={0} aria-pressed={selected}
      onClick={() => select(f.id)} onKeyDown={(e) => { if (e.key === "Enter" && e.target === e.currentTarget) onPlay(f); }}>
      <div className={s.fh}>
        <span className={`sev ${sc}`}>{sevLabel[sc]}</span>
        <span className={s.t}>{fmtClock(f.clipStartS ?? 0)}</span>
        {f.humanVerdict && <span className={`${s.state} ${s[f.humanVerdict]}`}>{f.humanVerdict}</span>}
        <span className={s.src}>{evalName(f.evaluator)}</span>
      </div>
      <div className={s.ft}>{f.title}</div>
      <div className={s.fe}>
        {f.evidence.length > 0 && !intendedHeard ? f.evidence.map((e, i) => <div key={i}><span className={s.q}>"{e.quote}"</span>{e.speaker ? <span className="faint"> {e.speaker}{e.startS != null ? `, ${fmtClock(e.startS)}` : ""}</span> : null}</div>) : null}
        <div className={f.evidence.length && !intendedHeard ? "faint" : ""} style={{ marginTop: f.evidence.length && !intendedHeard ? 3 : 0 }}>{f.detail}</div>
      </div>
      <div className={s.fx}>
        <Button size="sm" onClick={(e) => { e.stopPropagation(); onPlay(f); }} aria-label={`Play clip ${fmtClock(a)} to ${fmtClock(b)}`}><Icon name="play" /> clip {fmtClock(a)}-{fmtClock(b)} <Kbd>↵</Kbd></Button>
        <span className="grow" />
        <Button size="sm" tone="pass" aria-pressed={f.humanVerdict === "confirmed"} aria-label="Confirm finding" onClick={(e) => { e.stopPropagation(); onDecide(f, "confirmed"); }}><Icon name="check" /> <Kbd>y</Kbd></Button>
        <Button size="sm" aria-pressed={f.humanVerdict === "dismissed"} aria-label="Dismiss finding" onClick={(e) => { e.stopPropagation(); onDecide(f, "dismissed"); }}><Icon name="x" /> <Kbd>n</Kbd></Button>
        <Button size="sm" variant="ghost" title="Add note" onClick={(e) => { e.stopPropagation(); select(f.id); setNoteOpen(f.id); }}>note <Kbd>c</Kbd></Button>
      </div>
      {(noteOpen || f.humanNote) && (
        <div className={s.fnote} onClick={(e) => e.stopPropagation()}>
          {noteOpen ? (
            <textarea ref={ta} className="in" value={draft} placeholder="Why? One line is enough. Saved on blur or Ctrl+Enter." onChange={(e) => setDraft(e.target.value)} onBlur={save}
              onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); save(); } if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setNoteOpen(null); } }} />
          ) : (
            <div className={s.saved}>{f.humanNote} <a href="#" onClick={(e) => { e.preventDefault(); setNoteOpen(f.id); }}>edit</a></div>
          )}
        </div>
      )}
    </div>
  );
}

export function Rubric({ payload, onSeek }: { payload: ReviewPayload; onSeek: (t: number) => void }) {
  const crit = payload.eval?.criteria ?? [];
  if (!payload.evidence.scenario) return <div className="kvs faint" style={{ padding: "8px 12px", fontSize: 12 }}>No scenario selected, so there is no rubric to score. Pick one when uploading to enable it.</div>;
  if (!crit.length) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>Waiting for the transcript judge.</div>;
  return (
    <table className={s.rubric}><tbody>
      {crit.map((c) => (
        <tr key={c.criterionId}>
          <td><VerdictBadge v={c.status === "pass" ? "pass" : c.status === "fail" ? "fail" : "needs_review"} label={c.status === "unknown" ? "review" : undefined} /><div className={s.kind}>{c.kind.replace("_", " ")} {c.criterionId}</div></td>
          <td>{c.text}{c.evidence[0] ? <div className={s.q}>"{c.evidence[0].quote}" {c.evidence[0].startS != null && <a href="#" onClick={(e) => { e.preventDefault(); onSeek(c.evidence[0].startS!); }}>{fmtClock(c.evidence[0].startS)}</a>}</div> : c.reason ? <div className={s.q}>{c.reason}</div> : null}</td>
        </tr>
      ))}
    </tbody></table>
  );
}

export function DbDiff({ payload }: { payload: ReviewPayload }) {
  const sc = payload.call.scenarioSnapshot;
  const crit = payload.eval?.criteria.filter((c) => c.kind === "state") ?? [];
  if (!payload.evidence.dbSnapshots) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>Not available. This call has no database snapshots.</div>;
  const rows: { k: string; s: string }[] = [];
  for (const e of sc?.expectedState ?? []) {
    const r = crit.find((c) => c.criterionId === e.id);
    const where = Object.entries(e.where).map(([k, v]) => `${k}=${v}`).join(", ");
    const what = e.op === "row_added" ? `+1 row (${where})` : e.op === "no_change" ? `unchanged (${where})` : `${where}: ${Object.entries(e.changes).map(([k, v]) => `${k} -> ${v}`).join(", ")}`;
    rows.push({ k: "exp", s: `expected  ${e.table.padEnd(10)} ${what}` });
    rows.push(r?.status === "fail" ? { k: "miss", s: `actual    ${e.table.padEnd(10)} ${r.reason || "not as expected"}` } : { k: "ok", s: `actual    ${e.table.padEnd(10)} ${r ? "as expected" : "not checked yet"}` });
  }
  for (const c of payload.call.dbChanges) {
    const expected = (sc?.expectedState ?? []).some((e) => e.table === c.table);
    if (!expected) rows.push({ k: "unexp", s: `actual    ${c.table.padEnd(10)} ${c.kind} ${JSON.stringify(c.after ?? c.key)}  (not in expectations)` });
  }
  if (!rows.length) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>No expected changes and none observed.</div>;
  return <div className={s.dbdiff}>{rows.map((r, i) => <div key={i} className={s[r.k]}>{r.s}</div>)}</div>;
}

export function Evaluators({ payload, pending }: { payload: ReviewPayload; pending: boolean }) {
  const evs = payload.eval?.evaluators ?? [];
  const order = ["transcript_judge", "caller_validity", "state", "timing", "pronunciation", "audio_quality", "audio_listening"];
  const list = order.map((n) => evs.find((e) => e.name === n) ?? null);
  return (
    <table className={s.evals} style={{ width: "100%", borderCollapse: "collapse" }}><tbody>
      {order.map((n, i) => { const e = list[i]; return (
        <tr key={n}>
          <td style={{ width: 120 }}>{evalName(n)}{n === "audio_listening" && <div className={s.adv}>advisory</div>}</td>
          <td>{!e ? (pending ? <HealthDot state="loading">waiting</HealthDot> : <span className="faint">not run</span>) : e.status === "skipped" ? <span className="faint">Skipped. {e.reason}</span> : e.status === "error" ? <span style={{ color: "var(--fail)" }}>Error. {e.reason}</span> : <>{summarize(e.name, e.metrics)}</>}
            {e?.model && <div className="faint mono" style={{ fontSize: 10 }}>{e.model}</div>}</td>
        </tr>); })}
    </tbody></table>
  );
}

function summarize(name: string, m: Record<string, unknown>): string {
  const g = (k: string) => m[k] as number | undefined;
  switch (name) {
    case "transcript_judge": return g("criteriaTotal") != null ? `${g("criteriaPass")} of ${g("criteriaTotal")} criteria pass${g("unknown") ? `, ${g("unknown")} uncertain` : ""}` : "done";
    case "caller_validity": return g("checkpointsTotal") != null ? `${g("checkpointsReached")} of ${g("checkpointsTotal")} checkpoints` : "done";
    case "state": return g("expected") != null ? (g("missing") ? `${g("missing")} of ${g("expected")} expected changes missing` : "all expected changes present") : "done";
    case "timing": return g("reply_p50_s") != null ? `p50 reply ${g("reply_p50_s")} s, ${g("overlaps") ?? 0} overlaps, ${g("dead_air_events") ?? 0} dead-air events` : "done";
    case "pronunciation": return g("criticalTokens") != null ? `${g("mismatches")} critical token mismatch of ${g("criticalTokens")}` : "done";
    case "audio_quality": return g("mos") != null ? `MOS ${g("mos")}${g("baseline") != null ? ` (baseline ${g("baseline")})` : ""}${m.clipping ? ", clipping" : ""}` : "done";
    case "audio_listening": return g("notes") != null ? `${g("notes")} advisory note${g("notes") === 1 ? "" : "s"}` : "done";
    default: return "done";
  }
}

export function VerdictBar({ payload, human, onSet, disabled }: { payload: ReviewPayload; human: Verdict | null; onSet: (v: Verdict) => void; disabled?: boolean }) {
  const machine = payload.eval?.verdict ?? null;
  const invalid = machine === "invalid" && human !== "pass" && human !== "fail";
  return (
    <div className={s.verdictBar} id="verdictBar">
      <span className="muted">Call verdict</span>{machine ? <VerdictBadge v={machine} /> : <HealthDot state="loading">evaluating</HealthDot>}
      {human && human !== machine && <span className="faint">human: {verdictLabel[human]}</span>}
      {invalid && <span className="faint" title="Un-mark invalid first to count this call against the bot">invalid runs do not count against the bot</span>}
      <span className="grow" />
      <span className="seg" role="group" aria-label="Set human verdict">
        {(["pass", "fail", "needs_review", "invalid"] as Verdict[]).map((v, i) => <button key={v} type="button" disabled={disabled || (invalid && (v === "pass" || v === "fail"))} aria-pressed={(human ?? machine) === v} onClick={() => onSet(v)}><kbd>{i + 1}</kbd> {v === "needs_review" ? "review" : v}</button>)}
      </span>
    </div>
  );
}

export function selectFirstOpen(payload: ReviewPayload): string | null {
  const fs = payload.eval?.findings ?? [];
  return (fs.find((f) => !f.humanVerdict) ?? fs[0])?.id ?? null;
}

export function labelFor(payload: ReviewPayload, f: Finding) { return pinLabel(payload, f); }
