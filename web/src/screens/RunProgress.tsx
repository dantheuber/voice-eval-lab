import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, ErrorBox, ModeBadge, Panel, RateLegend, Skeleton, Untested, VerdictBadge } from "@/components/ui";
import { keys, useProfiles, useRun } from "@/api/queries";
import { isUntested } from "@/lib/untested";
import { endpoints } from "@/api/client";
import { useSocket } from "@/hooks/useSocket";
import { ciText, valid } from "@/lib/stats";
import { fmtClock, fmtDuration, fmtWhen } from "@/lib/format";
import type { CallSummary, RunDetail, RunWsMessage, Speaker } from "@/api/types";
import s from "./screens.module.css";

type Line = { kind: "turn" | "partial" | "event"; speaker?: Speaker; text: string; id: string };

export function RunProgressScreen() {
  const { id = "" } = useParams();
  const q = useRun(id);
  const profiles = useProfiles();
  const qc = useQueryClient();
  const nav = useNavigate();
  const [watch, setWatchState] = useState<string | null>(null);
  const watchRef = useRef<string | null>(null);
  const setWatch = (id: string | null) => { watchRef.current = id; setWatchState(id); };
  const [lines, setLines] = useState<Line[]>([]);
  const [liveT, setLiveT] = useState(0);
  const [connected, setConnected] = useState(false);
  const liveRef = useRef<HTMLDivElement>(null);
  const run = q.data?.run;
  const active = run?.status === "running" || run?.status === "paused" || run?.status === "queued";

  const patch = (fn: (d: RunDetail) => RunDetail) => qc.setQueryData<RunDetail>(keys.run(id), (prev) => (prev ? fn(prev) : prev));
  const upCall = (c: CallSummary) => patch((d) => ({ ...d, calls: d.calls.some((x) => x.id === c.id) ? d.calls.map((x) => (x.id === c.id ? c : x)) : [...d.calls, c] }));
  const open = useCallback(() => endpoints.runSocket(id), [id]);
  const sock = useSocket<RunWsMessage>(active ? open : null, (m, sk) => {
    setConnected(true);
    switch (m.type) {
      case "run": patch((d) => ({ ...d, run: m.run })); break;
      case "callStarted": upCall(m.call); if (!watch) { setWatch(m.call.id); setLines([]); sk.send(JSON.stringify({ type: "watch", callId: m.call.id })); } break;
      case "partial": if (m.callId === (watch ?? m.callId)) { setLiveT(m.atS); setLines((ls) => { const i = ls.findIndex((l) => l.kind === "partial" && l.speaker === m.speaker); const nl: Line = { kind: "partial", speaker: m.speaker, text: m.text, id: `p-${m.speaker}` }; return i >= 0 ? ls.map((l, j) => (j === i ? nl : l)) : [...ls, nl]; }); } break;
      case "turn": if (m.callId === (watch ?? m.callId)) { setLiveT(m.utterance.endS); setLines((ls) => [...ls.filter((l) => !(l.kind === "partial" && l.speaker === m.utterance.speaker)), { kind: "turn", speaker: m.utterance.speaker, text: m.utterance.heardText ?? m.utterance.intendedText ?? "", id: m.utterance.id }]); } break;
      case "event": if (m.callId === (watch ?? m.callId)) setLines((ls) => [...ls, { kind: "event", text: m.event.type === "tool_call" ? `${m.event.name}(${JSON.stringify(m.event.arguments)}) -> ${JSON.stringify(m.event.result)}` : m.event.type === "barge_in" ? `barge-in at ${fmtClock(m.event.atS)}${m.event.botStoppedAtS != null ? `, bot stopped in ${(m.event.botStoppedAtS - m.event.atS).toFixed(1)} s` : ""}` : m.event.detail, id: m.event.id }]); break;
      case "callDone": upCall(m.call); if (m.call.id === watch) { setWatch(null); } break;
      case "callEvaluated": upCall(m.call); qc.invalidateQueries({ queryKey: ["calls"] }); qc.invalidateQueries({ queryKey: ["overview"] }); break;
      case "runDone": patch((d) => ({ ...d, run: m.run })); qc.invalidateQueries({ queryKey: ["runs"] }); break;
    }
  }, {
    enabled: !!active, reconnect: true, onClose: () => setConnected(false),
    // After a reconnect, restore the watched call and refresh everything missed while offline.
    onOpen: (sk, attempt) => { if (watchRef.current) sk.send(JSON.stringify({ type: "watch", callId: watchRef.current })); if (attempt > 0) { qc.invalidateQueries({ queryKey: keys.run(id) }); qc.invalidateQueries({ queryKey: ["calls"] }); } },
  });
  useEffect(() => { liveRef.current?.scrollTo({ top: 1e9 }); }, [lines]);
  const watchCall = (c: CallSummary) => { setWatch(c.id); setLines([]); sock.current?.send(JSON.stringify({ type: "watch", callId: c.id })); };

  const act = async (a: "pause" | "resume" | "stop") => { const r = await endpoints.runAction(id, a); patch((d) => ({ ...d, run: r })); };

  return (
    <Shell crumbs={[<Link key="r" to="/runs/new">Run launcher</Link>, id]}>
      <div className="page">
        {q.isPending && <Skeleton rows={8} />}
        {q.isError && <ErrorBox error={q.error} retry={() => q.refetch()} what="this run" />}
        {q.data && run && (() => {
          const calls = q.data.calls;
          const running = calls.filter((c) => c.status === "running" || c.status === "evaluating");
          const watched = calls.find((c) => c.id === watch) ?? running[0] ?? null;
          const v = valid(run.counts);
          return (
            <>
              <div className="page-head">
                <div><h1 style={{ display: "flex", gap: 8, alignItems: "center" }}>{run.id} {run.status === "running" ? <VerdictBadge v="running" /> : <span className="tag">{run.status}</span>} <ModeBadge mode={run.mode} /></h1>
                  <div className="page-sub">{run.label}, <span className="mono">{run.profile}</span> {isUntested(run, profiles.data?.items) && <Untested />}, started {fmtWhen(run.startedAt)}.{run.status === "stopped" ? " Stopped early; partial results kept." : ""}{run.status === "paused" ? " Paused after the calls in flight." : ""}{active && !connected ? " Reconnecting to live updates…" : ""}</div></div>
                <div className="actions">
                  {run.status === "running" && <Button onClick={() => act("pause")}>Pause after current calls</Button>}
                  {run.status === "paused" && <Button onClick={() => act("resume")}>Resume</Button>}
                  {active && <Button variant="danger" onClick={() => { if (confirm("Stop this run? Calls in flight are marked failed.")) act("stop"); }}>Stop run</Button>}
                  {!active && <Link className="btn" to={`/calls?run=${run.id}`}>Open calls</Link>}
                  {!active && <Link className="btn" to={`/compare?a=${run.id}`}>Compare</Link>}
                </div>
              </div>
              <div className={s.progress}>
                <div className="stack">
                  <Panel>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}><span><b>{run.done}</b> of {run.calls} done{run.running ? `, ${run.running} running` : ""}</span><span className="faint">{run.etaS != null && active ? `about ${fmtDuration(run.etaS)} left` : run.endedAt ? `finished ${fmtWhen(run.endedAt)}` : ""}</span></div>
                    <div className={s.progBar} role="img" aria-label={`${run.done} of ${run.calls} calls done`}>{calls.map((c) => <i key={c.id} className={c.status === "running" || c.status === "evaluating" ? s.run : c.verdict === "pass" ? s.p : c.verdict === "fail" ? s.f : c.verdict === "needs_review" ? s.r : c.verdict === "invalid" ? s.i : ""} title={`${c.id} ${c.verdict ?? c.status}`} />)}</div>
                    <div style={{ marginTop: 8 }}><RateLegend counts={run.counts} /></div>
                    <div className="faint" style={{ marginTop: 8, fontSize: 11 }}>Pass rate so far <b>{v ? `${Math.round((100 * run.counts.pass) / v)}% (${ciText(run.counts.pass, v)}) of ${v} valid` : "n/a"}</b>.{active && v < 10 ? " With this few calls the interval is wide; wait for the run to finish before reading it." : ""}</div>
                  </Panel>
                  <Panel flush title="Calls" actions={<Link className="faint" to={`/calls?run=${run.id}`}>open list</Link>}>
                    <div style={{ maxHeight: 480, overflow: "auto" }}>
                      {calls.map((c) => (
                        <div key={c.id} className={s.callRow}>
                          <span>{c.status === "done" && c.verdict ? <VerdictBadge v={c.verdict} label=" " title={c.verdict} /> : <VerdictBadge v={c.status === "failed" ? "failed" : c.status === "queued" ? "queued" : "running"} label=" " title={c.status} />}</span>
                          <span>{c.scenarioTitle} <span className="muted">{c.personaName}</span> <span className="faint">rep {c.rep}</span><div className={s.id}>{c.id}{c.invalidReason ? ` · ${c.invalidReason}` : ""}</div></span>
                          <span>{c.status === "done" ? <Button size="sm" variant="ghost" onClick={() => nav(`/calls/${c.id}`)}>open</Button> : c.status === "running" ? <Button size="sm" variant="ghost" onClick={() => watchCall(c)} aria-pressed={watched?.id === c.id}>{watched?.id === c.id ? "watching" : "watch"}</Button> : <span className="faint" style={{ fontSize: 11 }}>{c.status}</span>}</span>
                        </div>
                      ))}
                    </div>
                  </Panel>
                </div>
                <div className="panel flush" style={{ display: "flex", flexDirection: "column", minHeight: 520 }}>
                  <div className="ph"><h2>Live call</h2>{watched ? <><span className="mono faint">{watched.id}</span><span className="tag">{watched.scenarioTitle}</span><span className="tag">{watched.personaName}</span></> : null}<span className="grow" /><span className="mono faint">{watched ? fmtClock(liveT) : ""}</span></div>
                  <div className={s.liveTurns} ref={liveRef} style={{ flex: 1, overflow: "auto" }}>
                    {!watched && <div className="empty"><b>{active ? "Waiting for the next call to start" : "Run finished"}</b>{active ? "The transcript streams here as the caller and bot speak." : "Open a call to review it."}</div>}
                    {watched && lines.length === 0 && <div className="faint" style={{ padding: 8 }}>Connecting to the call…</div>}
                    {lines.map((l) => l.kind === "event" ? <div key={l.id} className={`${s.lt} ${s.sys}`}><b>event</b><span>{l.text}</span></div> : <div key={l.id} className={`${s.lt} ${s[l.speaker!]}`}><b>{l.speaker}</b><span className={l.kind === "partial" ? "cursor" : ""}>{l.text}</span></div>)}
                  </div>
                  {running.filter((c) => c.id !== watched?.id).map((c) => <div key={c.id} className="ph" style={{ borderTop: "1px solid var(--line-1)", borderBottom: 0 }}><span className="faint">Also running: <span className="mono">{c.id}</span> {c.scenarioTitle}, {c.personaName}</span><span className="grow" /><Button size="sm" variant="ghost" onClick={() => watchCall(c)}>Watch instead</Button></div>)}
                </div>
              </div>
            </>
          );
        })()}
      </div>
    </Shell>
  );
}
