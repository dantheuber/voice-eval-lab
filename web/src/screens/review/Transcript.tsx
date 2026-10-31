import { useEffect, useRef } from "react";
import type { Finding, ReviewPayload, TimelineEvent, Utterance } from "@/api/types";
import { useReview } from "@/store/review";
import { evalName, fmtClock, sevClass } from "@/lib/format";
import { Icon } from "@/components/ui";
import s from "./review.module.css";

/* ASR picks its own punctuation, case, and hyphens, so nearly every real line differs in those. They aren't words the bot got wrong. */
const bare = (text: string) => text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
export const sameWords = (a: string, b: string) => bare(a) === bare(b);

/* Word-level diff good enough to highlight a substituted token. The pronunciation evaluator's alignment is used when present. */
export function diffWords(intended: string, heard: string): { pre: string; del: string; ins: string; post: string } {
  const a = intended.split(" "), b = heard.split(" ");
  let i = 0; while (i < a.length && i < b.length && bare(a[i]) === bare(b[i])) i++;
  let j = 0; while (j < a.length - i && j < b.length - i && bare(a[a.length - 1 - j]) === bare(b[b.length - 1 - j])) j++;
  return { pre: b.slice(0, i).join(" "), del: a.slice(i, a.length - j).join(" "), ins: b.slice(i, b.length - j).join(" "), post: b.slice(b.length - j).join(" ") };
}

const isUtt = (e: TimelineEvent): e is Utterance => e.type === "utterance";
const atOf = (e: TimelineEvent) => (isUtt(e) ? e.startS : e.atS);

export function findingsByUtterance(payload: ReviewPayload): Map<string, Finding[]> {
  const m = new Map<string, Finding[]>();
  const utts = payload.call.timeline.filter(isUtt);
  for (const f of payload.eval?.findings ?? []) {
    if (f.clipStartS == null) continue;
    // attach to the bot or caller utterance that overlaps the clip start most closely
    const cands = utts.filter((u) => u.startS <= (f.clipEndS ?? f.clipStartS!) && u.endS >= f.clipStartS!);
    const pick = cands.find((u) => u.speaker === "bot" && f.evaluator !== "timing") ?? cands[0];
    if (pick) m.set(pick.id, [...(m.get(pick.id) ?? []), f]);
  }
  return m;
}

export function Transcript({ payload, onSeek, onSelectFinding }: { payload: ReviewPayload; onSeek: (t: number) => void; onSelectFinding: (id: string) => void }) {
  const t = useReview((st) => st.t);
  const filter = useReview((st) => st.filter);
  const playing = useReview((st) => st.playing);
  const setFilter = useReview((st) => st.setFilter);
  const byUtt = findingsByUtterance(payload);
  const events = payload.call.timeline;
  const overlapSet = new Set<string>();
  const utts = events.filter(isUtt);
  for (const u of utts) if (u.speaker === "caller" && utts.some((b) => b.speaker === "bot" && b.startS < u.startS && b.endS > u.startS)) overlapSet.add(u.id);
  let activeId: string | null = null;
  for (const e of events) { if (atOf(e) <= t + 0.05) activeId = e.id; else break; }
  const activeRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (playing) activeRef.current?.scrollIntoView({ block: "center", behavior: "smooth" }); }, [activeId, playing]);

  const showIntended = payload.evidence.intendedText;
  return (
    <div className={s.transcript} id="transcript">
      <div className={s.filterbar}>
        <span className="seg" role="group" aria-label="Transcript filter">
          <button type="button" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>All turns</button>
          <button type="button" aria-pressed={filter === "findings"} onClick={() => setFilter("findings")}>With findings <kbd>f</kbd></button>
          <button type="button" aria-pressed={filter === "events"} onClick={() => setFilter("events")}>Tools and db</button>
        </span>
        <span className="grow" />
        <span className={`faint ${s.hintText}`}>{showIntended ? <>bot text shows <span className={s.del}>intended</span> <span className={s.ins}>heard</span> when they differ</> : "no intended text for this call; bot lines are what the ASR heard"}</span>
      </div>
      {events.length === 0 && <div className="empty"><b>No transcript</b>The call has no utterances yet.</div>}
      {events.map((e) => {
        const fs = isUtt(e) ? byUtt.get(e.id) ?? [] : [];
        if (filter === "findings" && !fs.length) return null;
        if (filter === "events" && isUtt(e)) return null;
        const active = e.id === activeId;
        if (!isUtt(e)) {
          if (e.type === "system" && e.kind !== "db_change") return null;
          const isDb = e.type === "system";
          const text = e.type === "tool_call" ? `${e.name}(${Object.entries(e.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})${e.error ? ` -> error: ${e.error}` : e.result != null ? ` -> ${typeof e.result === "string" ? e.result : JSON.stringify(e.result)}` : ""}` : e.type === "barge_in" ? `caller interrupted${e.botStoppedAtS != null ? `, bot stopped in ${(e.botStoppedAtS - e.atS).toFixed(1)} s` : ""}` : e.detail;
          return (
            <div key={e.id} ref={active ? activeRef : undefined} className={`${s.turn} ${s.event} ${active ? s.active : ""}`} onClick={() => onSeek(Math.max(0, e.atS - 0.5))}>
              <div className={s.who}>{e.type === "tool_call" ? "tool" : e.type === "barge_in" ? "barge" : "db"}<small>{fmtClock(e.atS)}</small></div>
              <div className={`${s.txt} ${isDb ? s.dbc : ""}`} style={e.type === "barge_in" ? { color: "var(--fail)" } : undefined}><Icon name={e.type === "tool_call" ? "tool" : isDb ? "db" : "warn"} size={12} /> {text}</div>
            </div>
          );
        }
        const heard = e.heardText ?? e.spokenText ?? "";
        const intended = e.intendedText;
        const differs = !!(showIntended && intended && e.speaker === "bot" && !sameWords(intended, heard) && !e.interrupted);
        const d = differs ? diffWords(intended!, heard) : null;
        return (
          <div key={e.id} ref={active ? activeRef : undefined} className={`${s.turn} ${s[e.speaker]} ${active ? s.active : ""}`} tabIndex={0} data-utt={e.id}
            onClick={() => { onSeek(e.startS); if (fs[0]) onSelectFinding(fs[0].id); }} onKeyDown={(ev) => { if (ev.key === "Enter") onSeek(e.startS); }}>
            <div className={s.who}>{e.speaker}<small>{fmtClock(e.startS)}</small></div>
            <div className={s.txt}>
              <div>
                {d ? <>{d.pre} <span className={s.del} title="intended">{d.del}</span> <span className={s.ins} title="heard by ASR">{d.ins}</span> {d.post}</> : heard || <span className="faint">(no speech recognised)</span>}
                {e.interrupted && <span className={s.barge}>interrupted</span>}
                {fs.map((f) => { const sc = sevClass(f.severity, f.advisory); return <span key={f.id} className={`${s.mark} ${s[sc]}`} onClick={(ev) => { ev.stopPropagation(); onSelectFinding(f.id); }}>{pinLabel(payload, f)} {evalName(f.evaluator)}</span>; })}
              </div>
              {(d || (e.interrupted && intended && intended !== heard)) && <div className={s.intended}><span className={s.lab}>intended</span>{intended}</div>}
              {overlapSet.has(e.id) && <div className={s.overlapNote}>overlaps bot speech</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function pinLabel(payload: ReviewPayload, f: Finding): string {
  const i = (payload.eval?.findings ?? []).findIndex((x) => x.id === f.id);
  return String(i + 1);
}
