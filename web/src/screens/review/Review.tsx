import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, ErrorBox, HealthDot, Icon, Kbd, ModeBadge, Note, Skeleton, Untested, VerdictBadge } from "@/components/ui";
import { keys, useCall, useDecideFinding, useProfiles, useSetVerdict } from "@/api/queries";
import { isUntested } from "@/lib/untested";
import { endpoints } from "@/api/client";
import { useSocket } from "@/hooks/useSocket";
import { useHotkeys, useScope } from "@/hooks/useHotkeys";
import { useReview } from "@/store/review";
import { toast } from "@/store/toast";
import { fmtClock, fmtDuration, sevClass, verdictLabel } from "@/lib/format";
import type { CallWsMessage, Finding, ReviewPayload, Verdict } from "@/api/types";
import { EventsLane } from "./EventsLane";
import { Waveform } from "./Waveform";
import { Transcript } from "./Transcript";
import { DbDiff, DeliveryPanel, Evaluators, FindingCard, Rubric, VerdictBar, clipRange, selectFirstOpen, verdictAllowed } from "./Findings";
import { useAudio } from "./useAudio";
import s from "./review.module.css";

const KEYS: [string, string][] = [
  ["Play or pause", "Space"], ["Play selected finding's clip", "Enter"], ["Next / previous finding", "j k"], ["Confirm finding", "y"], ["Dismiss finding", "n"], ["Add note to finding", "c"],
  ["Seek 2 s / 10 s", "← → Shift"], ["Jump to transcript turn", "t"], ["Next / previous call needing review", "] ["], ["Set call verdict pass / fail / review / invalid", "1 2 3 4"],
  ["Filter transcript to findings", "f"], ["Speed 1x / 1.5x / 2x", "s"], ["Mute caller / bot channel", "m b"],
];

export function ReviewScreen() {
  const { id = "" } = useParams();
  const q = useCall(id);
  return (
    <Shell crumbs={[<Link key="c" to="/calls">Calls</Link>, q.data?.call.batchId ? <Link key="r" to={`/calls?run=${q.data.call.batchId}`}>{q.data.call.batchId}</Link> : "…", id]} keys={KEYS}>
      {q.isPending && <div className="page"><Skeleton rows={8} /></div>}
      {q.isError && <div className="page"><ErrorBox error={q.error} retry={() => q.refetch()} what="this call" /></div>}
      {q.data && <ReviewBody key={id} payload={q.data} />}
    </Shell>
  );
}

function ReviewBody({ payload }: { payload: ReviewPayload }) {
  const id = payload.call.id;
  const nav = useNavigate();
  const qc = useQueryClient();
  const [sp, setSp] = useSearchParams();
  const st = useReview();
  const decide = useDecideFinding(id);
  const setVerdictM = useSetVerdict(id);
  const D = payload.call.durationS;
  const [showDetails, setShowDetails] = useState(false);
  const allFindings = useMemo(() => payload.eval?.findings ?? [], [payload.eval]);
  // Info findings are for someone already looking at the call, so they stay out of the way.
  const findings = useMemo(() => (showDetails ? allFindings : allFindings.filter((f) => f.severity !== "info")), [allFindings, showDetails]);
  const detailCount = allFindings.filter((f) => f.severity === "info").length;
  const audioSrc = endpoints.mediaUrl(payload.audio.available ? payload.audio.stereoUrl : null);
  const audio = useAudio(audioSrc, D);
  const failed = payload.call.status === "failed"; // a failed call is finished, and has no eval coming
  const pending = !failed && (payload.call.status === "running" || payload.call.status === "evaluating" || !payload.eval);
  const isTurn = payload.call.mode === "turn_based";
  const sc = payload.call.scenarioSnapshot;
  const pe = payload.call.personaSnapshot;
  const profiles = useProfiles();
  const untested = isUntested({ untested: payload.summary.untested, profile: payload.call.profile, providers: payload.call.providers }, profiles.data?.items);
  const guardOn = payload.call.metadata.argument_guard; // absent on realtime calls, which have no guard, and on older calls
  useScope("review");

  // live eval progress. Streamed messages patch the cache for immediacy; the final `eval` and any reconnect
  // refetch the whole payload so recording, timeline, duration, and timing are never left at their in-progress values.
  const openSock = useCallback(() => endpoints.callSocket(id), [id]);
  useSocket<CallWsMessage>(pending ? openSock : null, (m) => {
    qc.setQueryData<ReviewPayload>(keys.call(id), (prev) => {
      if (!prev) return prev;
      const ev = prev.eval ?? { callId: id, verdict: "needs_review" as Verdict, summary: "", criteria: [], findings: [], evaluators: [], completedAt: "" };
      if (m.type === "evaluator") return { ...prev, eval: { ...ev, evaluators: [...ev.evaluators.filter((e) => e.name !== m.evaluator.name), m.evaluator] } };
      if (m.type === "finding") return { ...prev, eval: { ...ev, findings: ev.findings.some((f) => f.id === m.finding.id) ? ev.findings : [...ev.findings, m.finding] } };
      if (m.type === "status" && m.status !== "done") return { ...prev, call: { ...prev.call, status: m.status }, summary: { ...prev.summary, status: m.status } };
      return prev;
    });
    if (m.type === "eval" || (m.type === "status" && m.status === "done") || m.type === "error") {
      qc.invalidateQueries({ queryKey: keys.call(id) });
      qc.invalidateQueries({ queryKey: ["calls"] }); qc.invalidateQueries({ queryKey: ["overview"] });
      if (m.type === "error") toast(`Evaluation error: ${m.message}`, 4000);
    }
  }, { enabled: pending, reconnect: true, onOpen: (_s, attempt) => { if (attempt > 0) qc.invalidateQueries({ queryKey: keys.call(id) }); } });

  // initial selection: URL finding, else first open
  useEffect(() => {
    st.setCall(id);
    const want = sp.get("finding");
    const first = (want && findings.some((f) => f.id === want) ? want : null) ?? selectFirstOpen(payload);
    if (first && !useReview.getState().selected) selectFinding(first, { cue: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, findings.length]);

  const selectFinding = (fid: string, opts: { cue?: boolean } = {}) => {
    const f = findings.find((x) => x.id === fid); if (!f) return;
    st.select(fid);
    const n = new URLSearchParams(sp); n.set("finding", fid); setSp(n, { replace: true });
    if (opts.cue) { const r = clipRange(f, D); st.setClip(r); st.seek(r[0]); }
    document.querySelector(`[data-finding="${fid}"]`)?.scrollIntoView({ block: "nearest" });
  };
  const playClip = (f: Finding) => { st.select(f.id); const [a, b] = clipRange(f, D); void audio.playRange(a, b); };
  const moveFinding = (dir: 1 | -1) => {
    if (!findings.length) return;
    const i = findings.findIndex((f) => f.id === st.selected);
    selectFinding(findings[(i + dir + findings.length) % findings.length].id, { cue: true });
  };
  const decideFinding = (f: Finding, v: "confirmed" | "dismissed") => {
    const next = f.humanVerdict === v ? null : v;
    decide.mutate({ findingId: f.id, humanVerdict: next, humanNote: f.humanNote });
    toast(next ? `Finding ${findings.indexOf(f) + 1} ${next}` : `Finding ${findings.indexOf(f) + 1} reopened`);
    if (next) {
      const nx = findings.find((x) => x.id !== f.id && !x.humanVerdict);
      if (nx) selectFinding(nx.id, { cue: true });
      else { toast("All findings reviewed. Set the call verdict (1-4) or press ] for the next call.", 3000); document.getElementById("verdictBar")?.scrollIntoView({ block: "end" }); }
    }
  };
  const saveNote = (f: Finding, note: string) => { if (note !== f.humanNote) { decide.mutate({ findingId: f.id, humanVerdict: f.humanVerdict, humanNote: note }); toast(note ? "Note saved" : "Note removed"); } };
  const setVerdict = (v: Verdict) => {
    if (!verdictAllowed(v, payload.eval?.verdict ?? null, payload.summary.humanVerdict, pending)) {
      toast(pending ? "Wait for evaluation to finish before setting a verdict." : "This run is invalid. Use \"Not invalid, count it\" (or 3) first.", 3000);
      return;
    }
    setVerdictM.mutate({ verdict: v, note: "" });
    toast(`Human verdict set to ${verdictLabel[v]}. Press ] for the next call.`, 2500);
  };
  const goNext = async () => { const n = payload.nextReviewId ?? (await endpoints.nextReview(id)).callId; if (n) nav(`/calls/${n}`); else toast("Nothing else needs review"); };
  const goPrev = () => { if (payload.prevReviewId) nav(`/calls/${payload.prevReviewId}`); else toast("No previous call"); };
  const seek = (t: number) => { st.setClip(null); st.seek(t); };

  const sel = findings.find((f) => f.id === st.selected) ?? null;
  useHotkeys({
    " ": () => void audio.toggle(),
    Enter: () => { if (sel) playClip(sel); },
    j: () => moveFinding(1), k: () => moveFinding(-1),
    y: () => sel && decideFinding(sel, "confirmed"), n: () => sel && decideFinding(sel, "dismissed"),
    c: () => { if (sel) st.setNoteOpen(sel.id); },
    ArrowLeft: () => seek(st.t - 2), ArrowRight: () => seek(st.t + 2), "Shift+ArrowLeft": () => seek(st.t - 10), "Shift+ArrowRight": () => seek(st.t + 10),
    f: () => st.setFilter(st.filter === "findings" ? "all" : "findings"), s: () => st.cycleSpeed(),
    m: () => st.toggleMute("caller"), b: () => st.toggleMute("bot"),
    t: () => (document.querySelector(`.${s.turn}.${s.active}`) as HTMLElement | null)?.focus(),
    "]": () => void goNext(), "[": goPrev,
    "1": () => setVerdict("pass"), "2": () => setVerdict("fail"), "3": () => setVerdict("needs_review"), "4": () => setVerdict("invalid"),
    Escape: () => { if (st.noteOpen) st.setNoteOpen(null); },
  }, { scope: "review" });

  const openCount = findings.filter((f) => !f.humanVerdict && !f.advisory).length;
  const selIndex = sel ? findings.indexOf(sel) + 1 : 0;
  const invalidReason = payload.summary.invalidReason;
  const human = payload.summary.humanVerdict;
  const ticks = useMemo(() => { const step = D > 240 ? 30 : 15; const out: number[] = []; for (let t = 0; t <= D; t += step) out.push(t); return out; }, [D]);
  const lanesRef = useRef<HTMLDivElement>(null);

  return (
    <div className={s.review}>
      <div className={s.head}>
        <div className={s.title}><span className="mono">{id}</span>{payload.eval ? <VerdictBadge v={payload.eval.verdict} /> : <VerdictBadge v={failed ? "failed" : payload.call.status === "running" ? "running" : "evaluating"} />}<ModeBadge mode={payload.call.mode} /></div>
        <div className={s.meta}>
          <span>{sc ? <><b>{sc.title}</b></> : <span className="faint">no scenario</span>}</span>
          <span>persona <b>{pe?.name ?? (payload.call.mode === "human" ? "you" : "unknown")}</b></span>
          <span>profile <b className="mono">{payload.call.profile ?? "?"}</b> {untested && <Untested />}</span>
          {typeof guardOn === "boolean" && <span title="The support bot's check on tool arguments. Off runs the raw model, so a made-up value reaches the tool.">argument guard <b>{guardOn ? "on" : "off"}</b></span>}
          {payload.call.batchId && <span>{payload.call.batchId}{payload.summary.rep ? <>, rep <b>{payload.summary.rep}</b></> : null}</span>}
          <span><b>{fmtClock(D)}</b> call time{isTurn && payload.call.endedAt && payload.call.startedAt ? <>, <b>{fmtDuration((new Date(payload.call.endedAt).getTime() - new Date(payload.call.startedAt).getTime()) / 1000)}</b> wall clock</> : null}</span>
          {sc && <span title="caller turns used of the scenario's limit">{payload.call.timeline.filter((e) => e.type === "utterance" && e.speaker === "caller").length}/{sc.maxTurns} turns</span>}
        </div>
        <div className={s.nav}>
          <span className="faint">{findings.length ? `finding ${selIndex} of ${findings.length}` : pending ? "evaluating" : "no findings"}</span>
          <Button size="sm" onClick={goPrev} disabled={!payload.prevReviewId} title="Previous call needing review"><Kbd>[</Kbd> prev</Button>
          <Button size="sm" variant="primary" onClick={goNext} title="Next call needing review">next needing review <Kbd>]</Kbd></Button>
        </div>
        {isTurn && <Note kind="warn" style={{ flexBasis: "100%" }}><b>Turn-based call.</b> Each side waited for the other to finish, so the gaps you see are synthetic and timing was not measured. Transcript, tools, database changes, pronunciation, and rubric results are real.</Note>}
        {payload.eval?.verdict === "invalid" && <Note kind="plain" style={{ flexBasis: "100%" }}><b>Invalid run.</b> {invalidReason ?? "The caller never reached the scenario."} This is a test problem, not a bot failure, and the call is excluded from pass rates.</Note>}
        {payload.call.mode === "upload" && !payload.evidence.toolLog && <Note kind="plain" style={{ flexBasis: "100%" }}><b>Reduced evidence.</b> No tool log{!payload.evidence.intendedText ? ", no intended text" : ""}{!payload.evidence.scenario ? ", no scenario" : ""}. Checks that need them were skipped and the call cannot pass or fail automatically.</Note>}
        {payload.call.error && <Note kind="err" style={{ flexBasis: "100%" }}><b>Call failed.</b> {payload.call.error}</Note>}
      </div>
      <div className={s.body}>
        <section className={s.timeline} aria-label="Timeline">
          <div className={s.toolbar}>
            <Button size="sm" onClick={() => void audio.toggle()} aria-label={st.playing ? "Pause" : "Play"} disabled={!payload.audio.available}><Icon name={st.playing ? "pause" : "play"} /> <Kbd>Space</Kbd></Button>
            <span className={s.time}>{fmtClock(st.t, true)} <span>/ {fmtClock(D, true)}</span></span>
            <Button size="sm" variant="ghost" onClick={st.cycleSpeed} title="Playback speed">{st.speed}x</Button>
            <span className="faint">{st.clip ? `clip ${fmtClock(st.clip[0])} to ${fmtClock(st.clip[1])}${sel ? `, finding ${selIndex}` : ""}` : ""}</span>
            <span className="grow" />
            <span className={s.legend}><span><i style={{ background: "var(--caller)" }} />caller</span><span><i style={{ background: "var(--bot)" }} />bot</span><span><i className={s.hatch} />overlap</span><span><i style={{ background: "var(--accent-soft)", border: "1px solid var(--accent)" }} />clip</span></span>
            {isTurn ? <span className="tag" title="Timing evaluator skipped">gaps synthetic</span> : payload.timing ? <span className="tag mono" title="median bot reply latency">p50 {payload.timing.replyP50S}s</span> : null}
          </div>
          <div className={s.wrap}>
            {payload.audio.available && payload.peaks ? (
              <div className={s.lanes} ref={lanesRef}>
                <div className={s.ln}>findings</div>
                <div className={s.findingsLane}>{findings.map((f, i) => { const sc2 = sevClass(f.severity, f.advisory); return <button key={f.id} type="button" className={`${s.pin} ${s[sc2]} ${f.humanVerdict ? s.done : ""} ${st.selected === f.id ? s.sel : ""}`} style={{ left: `${((f.clipStartS ?? 0) / D) * 100}%` }} title={f.title} aria-label={`Finding ${i + 1} at ${fmtClock(f.clipStartS ?? 0)}: ${f.title}`} onClick={() => playClip(f)}>{i + 1}</button>; })}</div>
                <div className={`${s.ln} ${s.caller}`}><button type="button" aria-pressed={st.mute.caller} title="Mute caller channel (m)" onClick={() => st.toggleMute("caller")}><i />caller</button></div>
                <div className={`${s.lane} ${s.caller} ${st.mute.caller ? s.muted : ""}`}><Waveform peaks={payload.peaks.caller} perSecond={payload.peaks.perSecond} durationS={D} overlaps={payload.overlaps} color="caller" onSeek={seek} /></div>
                <div className={`${s.ln} ${s.bot}`}><button type="button" aria-pressed={st.mute.bot} title="Mute bot channel (b)" onClick={() => st.toggleMute("bot")}><i />bot</button></div>
                <div className={`${s.lane} ${s.bot} ${st.mute.bot ? s.muted : ""}`}><Waveform peaks={payload.peaks.bot} perSecond={payload.peaks.perSecond} durationS={D} overlaps={payload.overlaps} color="bot" onSeek={seek} /></div>
                <div className={s.ln}>events</div>
                <EventsLane timeline={payload.call.timeline} durationS={D} onSeek={seek} />
                <div className={s.ln} />
                <div className={s.ruler}>{ticks.map((t) => <span key={t} style={{ left: `${(t / D) * 100}%` }}>{fmtClock(t)}</span>)}</div>
                <Overlay lanesRef={lanesRef} D={D} />
              </div>
            ) : (
              <div className={s.unavailable}>Recording unavailable{payload.audio.stereoUrl ? ` (${payload.audio.stereoUrl})` : ""}. Transcript and findings are still shown; clip playback is disabled.</div>
            )}
          </div>
        </section>
        <Transcript payload={payload} onSeek={seek} onSelectFinding={(fid) => selectFinding(fid)} />
        <aside className={s.side} aria-label="Findings and evaluators">
          <div className={s.sec} id="secFindings">
            <div className={s.sh}>Findings <span className="tag mono">{openCount} open</span>{pending && <HealthDot state="loading">evaluating</HealthDot>}<span className="grow" /><span className={s.hint}><kbd>j</kbd><kbd>k</kbd> move, <kbd>y</kbd> confirm, <kbd>n</kbd> dismiss</span></div>
            {findings.length === 0 && !pending && <div className="empty" style={{ padding: 16 }}>{failed ? <><b>Not evaluated</b>The call failed before it finished, so nothing was checked. What was recorded is above.</> : <><b>No findings</b>Every evaluator passed this call. Set the verdict or press <kbd>]</kbd>.</>}</div>}
            {findings.length === 0 && pending && <div style={{ padding: 12 }}><Skeleton rows={3} /></div>}
            {findings.map((f) => <FindingCard key={f.id} payload={payload} f={f} onPlay={playClip} onDecide={decideFinding} onSaveNote={saveNote} />)}
            {detailCount > 0 && <div style={{ padding: "6px 12px" }}><Button size="sm" variant="ghost" onClick={() => setShowDetails(!showDetails)}>{showDetails ? "Hide" : "Show"} {detailCount} detail{detailCount === 1 ? "" : "s"}</Button></div>}
          </div>
          <div className={s.sec}><div className={s.sh}>Rubric{sc ? `, ${sc.title}` : ""}</div><Rubric payload={payload} onSeek={seek} pending={pending} /></div>
          <div className={s.sec}><div className={s.sh}>Database changes</div><DbDiff payload={payload} /></div>
          <div className={s.sec}><div className={s.sh}>Timing</div>
            {payload.timing ? <dl className={s.kvs}><dt>reply p50 / p95</dt><dd>{payload.timing.replyP50S} s / {payload.timing.replyP95S} s</dd><dt>overlaps</dt><dd>{payload.timing.overlaps}</dd><dt>dead air events</dt><dd>{payload.timing.deadAirEvents}</dd><dt>barge-in stop time</dt><dd>{payload.timing.bargeStopS != null ? `${payload.timing.bargeStopS} s` : "no barge-ins"}</dd></dl>
              : <dl className={s.kvs}><dt>reply latency</dt><dd className={s.na}>{isTurn ? "not measured (turn-based)" : pending ? "waiting" : "not measured"}</dd><dt>overlaps</dt><dd className={s.na}>not measured</dd><dt>dead air</dt><dd className={s.na}>not measured</dd></dl>}
          </div>
          <div className={s.sec}><div className={s.sh}>Delivery, measured <span className="tag" title="Advisory. Limits come from a small corpus, no listener ratings.">advisory</span></div><DeliveryPanel payload={payload} onSeek={seek} /></div>
          <div className={s.sec}><div className={s.sh}>Evaluators</div><Evaluators payload={payload} pending={pending} /></div>
          <VerdictBar payload={payload} human={human} onSet={setVerdict} pending={pending} />
        </aside>
      </div>
    </div>
  );
}

/* Playhead and clip band positioned over the lane column. */
function Overlay({ lanesRef, D }: { lanesRef: React.RefObject<HTMLDivElement | null>; D: number }) {
  const t = useReview((st) => st.t);
  const clip = useReview((st) => st.clip);
  const el = lanesRef.current;
  const lane = el?.querySelector(`.${s.lane}`) as HTMLElement | null;
  if (!el || !lane) return null;
  const left = lane.offsetLeft, width = lane.offsetWidth;
  const x = (tt: number) => left + (Math.min(D, Math.max(0, tt)) / D) * width;
  return (
    <>
      {clip && <div className={s.clip} style={{ left: x(clip[0]), width: x(clip[1]) - x(clip[0]), top: 0, bottom: 18, position: "absolute" }} />}
      <div className={s.playhead} style={{ left: x(t), position: "absolute", top: 0, bottom: 18 }} />
    </>
  );
}
