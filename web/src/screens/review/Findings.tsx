import { useEffect, useRef, useState } from "react";
import type { Finding, ReviewPayload, Verdict } from "@/api/types";
import { Button, HealthDot, Icon, Kbd, VerdictBadge } from "@/components/ui";
import { evalName, fmtClock, sevClass, sevLabel, verdictLabel } from "@/lib/format";
import { categoryLabel, categoryNote, deliveryAt, deliveryFor, deliveryLine, isAdvisoryEvaluator, orderedEvaluators, summarizeEvaluator, type DeliveryMetrics } from "@/lib/evaluators";
import { useReview } from "@/store/review";
import s from "./review.module.css";
import { pinLabel } from "./Transcript";

export const clipRange = (f: Finding, durationS: number): [number, number] => [Math.max(0, (f.clipStartS ?? 0) - 1), Math.min(durationS, (f.clipEndS ?? (f.clipStartS ?? 0) + 5) + 1)];

export interface Moment { startS: number; endS: number }

/** Each moment a rolled-up finding covers, in call order. One moment is just the finding's clip. */
export function moments(f: Finding): Moment[] {
  const timed = f.evidence.filter((e) => e.startS != null).map((e) => ({ startS: e.startS!, endS: e.endS ?? e.startS! }));
  return timed.length > 1 ? timed.sort((a, b) => a.startS - b.startS) : [];
}

export const momentClip = (f: Finding, m: Moment): Finding => ({ ...f, clipStartS: m.startS, clipEndS: m.endS });

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
  const catLabel = categoryLabel(f), catNote = categoryNote(f);
  // An audio listener's claim is checked against the measured delivery of the same utterance; the measurement wins.
  const measured = f.evaluator === "audio_listening" ? deliveryAt(deliveryFor(payload.eval?.evaluators), f.clipStartS, f.clipEndS) : null;
  return (
    <div className={`${s.finding} ${selected ? s.sel : ""} ${f.advisory ? s.advisory : ""}`} data-finding={f.id} role="button" tabIndex={0} aria-pressed={selected}
      onClick={() => select(f.id)} onKeyDown={(e) => { if (e.key === "Enter" && e.target === e.currentTarget) onPlay(f); }}>
      <div className={s.fh}>
        <span className={`sev ${sc}`}>{sevLabel[sc]}</span>
        <span className={s.t}>{fmtClock(f.clipStartS ?? 0)}</span>
        {f.humanVerdict && <span className={`${s.state} ${s[f.humanVerdict]}`}>{f.humanVerdict}</span>}
        <span className={s.src}>{evalName(f.evaluator)}{catLabel && catLabel !== evalName(f.evaluator) ? <> · <span title={catNote}>{catLabel}</span></> : null}</span>
      </div>
      <div className={s.ft}>{f.title}</div>
      <div className={s.fe}>
        {f.evidence.filter((e) => e.quote).length > 0 && !intendedHeard ? f.evidence.filter((e) => e.quote).map((e, i) => <div key={i}><span className={s.q}>"{e.quote}"</span>{e.speaker ? <span className="faint"> {e.speaker}{e.startS != null ? `, ${fmtClock(e.startS)}` : ""}</span> : null}</div>) : null}
        <div className={f.evidence.some((e) => e.quote) && !intendedHeard ? "faint" : ""} style={{ marginTop: f.evidence.some((e) => e.quote) && !intendedHeard ? 3 : 0 }}>{f.detail}</div>
        {catNote && <div className="faint" style={{ marginTop: 3 }}>{catNote}</div>}
        {moments(f).length > 0 && <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>{moments(f).map((m) => <Button key={m.startS} size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); onPlay(momentClip(f, m)); }} aria-label={`Play the occurrence at ${fmtClock(m.startS)}`}><Icon name="play" /> {fmtClock(m.startS)}</Button>)}</div>}
        {measured && <div style={{ marginTop: 3 }}><span className="faint">Measured on this utterance:</span> <code>{deliveryLine(measured)}</code>{measured.flags.length ? <span className="faint">, flagged {measured.flags.map((x) => categoryLabel({ category: x })).join(", ")}</span> : <span className="faint">, no delivery flags. The measurement wins over the listener's impression.</span>}</div>}
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

export function Rubric({ payload, onSeek, pending }: { payload: ReviewPayload; onSeek: (t: number) => void; pending: boolean }) {
  const crit = payload.eval?.criteria ?? [];
  if (!payload.evidence.scenario) return <div className="kvs faint" style={{ padding: "8px 12px", fontSize: 12 }}>No scenario selected, so there is no rubric to score. Pick one when uploading to enable it.</div>;
  if (!crit.length) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>{pending ? "Waiting for the transcript judge." : "Not judged."}</div>;
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
    rows.push(dbActualRow(e.table, r?.status ?? null, r?.reason ?? ""));
  }
  for (const c of payload.call.dbChanges) {
    const expected = (sc?.expectedState ?? []).some((e) => e.table === c.table);
    if (!expected) rows.push({ k: "unexp", s: `actual    ${c.table.padEnd(10)} ${c.kind} ${JSON.stringify(c.after ?? c.key)}  (not in expectations)` });
  }
  if (!rows.length) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>No expected changes and none observed.</div>;
  return <div className={s.dbdiff}>{rows.map((r, i) => <div key={i} className={s[r.k]}>{r.s}</div>)}</div>;
}

/* Only a pass may claim success. Unknown means the evaluator could not decide; absent means it has not run. */
export function dbActualRow(table: string, status: "pass" | "fail" | "unknown" | null, reason: string): { k: "ok" | "miss" | "unexp" | "exp"; s: string } {
  const head = `actual    ${table.padEnd(10)} `;
  if (status === "pass") return { k: "ok", s: `${head}as expected${reason ? ` (${reason})` : ""}` };
  if (status === "fail") return { k: "miss", s: `${head}${reason || "not as expected"}` };
  if (status === "unknown") return { k: "unexp", s: `${head}could not be checked${reason ? `: ${reason}` : ""}` };
  return { k: "exp", s: `${head}not checked yet` };
}

/* Which human verdicts are allowed right now. Shared by the buttons and the hotkeys. */
export function verdictAllowed(v: Verdict, machine: Verdict | null, human: Verdict | null, pending: boolean): boolean {
  if (pending) return false;
  const invalidLocked = machine === "invalid" && (human == null || human === "invalid");
  if (invalidLocked && (v === "pass" || v === "fail")) return false;
  return true;
}

/* Driven by the server's evaluator list, so a new evaluator shows up without a frontend change. */
export function Evaluators({ payload, pending }: { payload: ReviewPayload; pending: boolean }) {
  const evs = payload.eval?.evaluators ?? [];
  const findings = payload.eval?.findings ?? [];
  const names = orderedEvaluators(evs, findings, pending);
  return (
    <table className={s.evals} style={{ width: "100%", borderCollapse: "collapse" }}><tbody>
      {names.map((n) => { const e = evs.find((x) => x.name === n) ?? null; return (
        <tr key={n}>
          <td style={{ width: 120 }}>{evalName(n)}{isAdvisoryEvaluator(n, findings) && <div className={s.adv}>advisory</div>}</td>
          <td>{!e ? (pending ? <HealthDot state="loading">waiting</HealthDot> : <span className="faint">not run</span>) : e.status === "skipped" ? <span className="faint">Skipped. {e.reason}</span> : e.status === "error" ? <span style={{ color: "var(--fail)" }}>Error. {e.reason}</span> : <>{summarizeEvaluator(e.name, e.metrics)}</>}
            {e?.model && <div className="faint mono" style={{ fontSize: 10 }}>{e.model}</div>}</td>
        </tr>); })}
    </tbody></table>
  );
}

/* Per-utterance delivery measurements for the bot: rate, pitch movement, pauses, noise. Advisory; it checks the audio listener. */
export function DeliveryPanel({ payload, onSeek }: { payload: ReviewPayload; onSeek: (t: number) => void }) {
  const run = payload.eval?.evaluators.find((r) => r.name === "delivery");
  if (!run) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>Not run on this call.</div>;
  if (run.status !== "ok") return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>{run.status === "skipped" ? "Skipped." : "Error."} {run.reason}</div>;
  const d: DeliveryMetrics | null = deliveryFor([run]);
  if (!d) return <div className="faint" style={{ padding: "8px 12px", fontSize: 12 }}>No per-utterance metrics in this result.</div>;
  const fmt = (v: number | null, digits = 0, unit = "") => (v == null ? <span className="faint">–</span> : `${v.toFixed(digits)}${unit}`);
  return (
    <div>
      <div className="faint" style={{ padding: "6px 12px 0", fontSize: 11 }}>Bot only, measured from the signal. Medians: {d.medians.wpm != null ? `${Math.round(d.medians.wpm)} wpm` : "no rate"}, pitch IQR {d.medians.pitchIqrSt != null ? `${d.medians.pitchIqrSt.toFixed(1)} st` : "–"}, SNR {d.medians.snrDb != null ? `${Math.round(d.medians.snrDb)} dB` : "–"}.{d.wpmNote ? ` ${d.wpmNote}.` : ""} Where this disagrees with the audio listener, the measurement wins.</div>
      <table className={s.delivery}>
        <thead><tr><th>at</th><th className="num" title="words per minute of speech time">wpm</th><th className="num" title="pitch interquartile range in semitones, median Hz in the title">pitch</th><th className="num" title="pause count and longest pause">pauses</th><th className="num" title="signal to noise ratio, 300 to 3400 Hz">SNR</th><th>flags</th></tr></thead>
        <tbody>{d.bot.map((u, i) => (
          <tr key={u.utteranceId ?? i} onClick={() => onSeek(u.startS)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSeek(u.startS)} className={u.flags.length ? s.flagged : ""}>
            <td className="mono">{fmtClock(u.startS)}</td>
            <td className="num mono">{fmt(u.wpm)}</td>
            <td className="num mono" title={u.pitchMedianHz != null ? `median ${Math.round(u.pitchMedianHz)} Hz, range ${u.pitchRangeSt?.toFixed(1) ?? "–"} st` : "not enough voiced speech"}>{fmt(u.pitchIqrSt, 1, " st")}</td>
            <td className="num mono" title={`${Math.round(u.pauseRatio * 100)}% of the utterance was silence`}>{u.pauseCount}{u.pauseCount ? <span className="faint"> / {u.longestPauseS.toFixed(1)} s</span> : null}</td>
            <td className="num mono">{fmt(u.snrDb, 0, " dB")}</td>
            <td className="faint">{u.flags.map((x) => categoryLabel({ category: x })).join(", ")}{u.chops ? `${u.flags.length ? ", " : ""}${u.chops} chop${u.chops === 1 ? "" : "s"}` : ""}</td>
          </tr>))}</tbody>
      </table>
      {d.limitation && <div className="faint" style={{ padding: "4px 12px 8px", fontSize: 10 }}>{d.limitation}</div>}
    </div>
  );
}

export function VerdictBar({ payload, human, onSet, pending }: { payload: ReviewPayload; human: Verdict | null; onSet: (v: Verdict) => void; pending: boolean }) {
  const machine = payload.eval?.verdict ?? null;
  const invalidLocked = machine === "invalid" && (human == null || human === "invalid");
  return (
    <div className={s.verdictBar} id="verdictBar">
      <span className="muted">Call verdict</span>{machine ? <VerdictBadge v={machine} /> : pending ? <HealthDot state="loading">evaluating</HealthDot> : <VerdictBadge v="failed" />}
      {human && human !== machine && <span className="faint">human: {verdictLabel[human]}</span>}
      {invalidLocked && <span className="faint">invalid runs do not count against the bot. <button type="button" className="btn ghost sm" onClick={() => onSet("needs_review")} title="Mark this run as a real test of the bot; pass and fail unlock">Not invalid, count it</button></span>}
      <span className="grow" />
      <span className="seg" role="group" aria-label="Set human verdict">
        {(["pass", "fail", "needs_review", "invalid"] as Verdict[]).map((v, i) => <button key={v} type="button" disabled={!verdictAllowed(v, machine, human, pending)} aria-pressed={(human ?? machine) === v} onClick={() => onSet(v)}><kbd>{i + 1}</kbd> {v === "needs_review" ? "review" : v}</button>)}
      </span>
    </div>
  );
}

export function selectFirstOpen(payload: ReviewPayload): string | null {
  const fs = payload.eval?.findings ?? [];
  return (fs.find((f) => !f.humanVerdict) ?? fs[0])?.id ?? null;
}

export function labelFor(payload: ReviewPayload, f: Finding) { return pinLabel(payload, f); }
