import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Field, HealthDot, Icon, Kbd, ModeBadge, Note, Panel, Seg, VerdictBadge } from "@/components/ui";
import { useHealth, usePersonas, useProfiles, useScenarios } from "@/api/queries";
import { endpoints } from "@/api/client";
import { useHotkeys } from "@/hooks/useHotkeys";
import { useSocket } from "@/hooks/useSocket";
import { toast } from "@/store/toast";
import { evalName, fmtClock } from "@/lib/format";
import { isAdvisoryEvaluator, orderedEvaluators } from "@/lib/evaluators";
import { isMock } from "@/api/transport";
import type { CallWsMessage, EvaluatorRun, Finding, LineConditions, TalkSession } from "@/api/types";
import { useMic } from "./useMic";
import { useBotCall } from "./useBotCall";
import s from "./talk.module.css";

// What a mic call needs. A realtime profile has "realtime" instead of an LLM, and "off" means the profile doesn't use that role.
const TALK_ROLES = ["llm_support", "stt_live", "tts_bot", "realtime"];

const LINES: Record<string, LineConditions | null> = {
  none: null,
  phone: { codec: "mulaw_8k", noise: "none", snrDb: 22, packetLossPct: 0, oneWayLatencyMs: 80 },
  noisy: { codec: "mulaw_8k", noise: "car", snrDb: 12, packetLossPct: 2, oneWayLatencyMs: 120 },
};

export function TalkScreen() {
  const [sp] = useSearchParams();
  const [phase, setPhase] = useState<"setup" | "incall" | "evaluating">("setup");
  const [session, setSession] = useState<TalkSession | null>(null);
  const [endedCallId, setEndedCallId] = useState<string | null>(null);
  const scenarios = useScenarios();
  const personas = usePersonas();
  const health = useHealth();
  const profiles = useProfiles();
  const [scenarioId, setScenarioId] = useState<string>(sp.get("scenario") ?? "");
  const [personaId, setPersonaId] = useState<string>("");
  const [line, setLine] = useState<"none" | "phone" | "noisy">("none");
  const [record, setRecord] = useState(true);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const mic = useMic(deviceId, phase !== "evaluating");
  const levelRef = useRef(0);
  levelRef.current = mic.level;
  const botAudio = useRef<HTMLAudioElement | null>(null);
  const call = useBotCall(botAudio, levelRef);
  const notReady = health.data?.items.filter((h) => TALK_ROLES.includes(h.role) && (h.state === "loading" || h.state === "down")) ?? [];

  const create = useMutation({
    mutationFn: () => endpoints.createTalkSession({ scenarioId: scenarioId || null, personaId: personaId || null, line: LINES[line], record, profile: profiles.data?.activeId ?? null }),
    onSuccess: async (sess) => { setSession(sess); setPhase("incall"); await call.start(sess, { deviceId }); },
  });
  const endM = useMutation({
    mutationFn: async () => { call.end(); const r = await endpoints.endTalkSession(session!.sessionId); return r.callId; },
    onSuccess: (callId) => {
      if (!record) { toast("Call ended. Unlisted calls aren't kept."); setPhase("setup"); setSession(null); return; }
      setEndedCallId(callId); setPhase("evaluating");
    },
  });
  // The bot can hang up and the connection can drop, and then there's nothing left to wait for on this screen.
  const hungUp = call.state.status === "ended";
  useEffect(() => { if (phase === "incall" && hungUp && !endM.isPending && !endM.isError) endM.mutate(); }, [phase, hungUp, endM]);
  const escOnce = useRef(false);
  useHotkeys({
    Enter: () => { if (phase === "setup" && !create.isPending && !mic.error) create.mutate(); },
    Escape: () => { if (phase !== "incall") return; if (escOnce.current) endM.mutate(); else { escOnce.current = true; toast("Press Esc again to end the call"); setTimeout(() => { escOnce.current = false; }, 1500); } },
    m: () => { if (phase === "incall") call.setMuted(!call.state.muted); },
  });
  useEffect(() => { if (call.state.status === "error" && phase === "incall") toast(call.state.error ?? "Call error", 4000); }, [call.state.status, call.state.error, phase]);

  const nav = useNavigate();
  return (
    <Shell crumbs={["Talk to the bot"]} keys={[["Start call", "Enter"], ["End call", "Esc Esc"], ["Mute mic", "m"]]}>
      <audio ref={botAudio} autoPlay playsInline style={{ display: "none" }} />
      <div className="page" style={{ height: "100%" }}>
        {phase === "setup" && (
          <div className={s.talk}>
            <div className="stack">
              <Panel title="Before you call">
                <div className="stack" style={{ gap: 10 }}>
                  <Field label="Microphone"><select className="in" value={deviceId ?? ""} onChange={(e) => setDeviceId(e.target.value || null)} disabled={!mic.devices.length}><option value="">{mic.devices.length ? "System default" : mic.error ? "unavailable" : "requesting permission…"}</option>{mic.devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId.slice(0, 8)}</option>)}</select></Field>
                  <Field label="Level" help={mic.error ? undefined : "Say something. Bars should reach the middle when you speak normally."}><Meter level={mic.level} who="you" /></Field>
                  {mic.error && <Note kind="err">{mic.error}</Note>}
                  <Note kind="info"><Icon name="headphones" /> <b>Headphones recommended.</b>{mic.headphonesLikely ? " Headphones detected." : " On speakers the bot may hear itself."} Echo cancellation is on, gain control is off, and the server ignores echo-like audio, but interruption timing is less reliable on speakers.</Note>
                  <Field label="Record as"><Seg value={record ? "mic" : "unlisted"} options={[{ value: "mic", label: <><Icon name="talk" /> mic call</> }, { value: "unlisted", label: "unlisted" }]} onChange={(v) => setRecord(v === "mic")} /></Field>
                </div>
              </Panel>
            </div>
            <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
              <div className="ph"><h2>What are you calling about?</h2></div>
              <div className="pb" style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div><div className="label" style={{ marginBottom: 6 }}>Play a scenario (your facts and beats are shown during the call)</div>
                  <div className={s.pick}>
                    {scenarios.data?.items.map((sc) => <label key={sc.id}><input type="radio" name="what" checked={scenarioId === sc.id} onChange={() => setScenarioId(sc.id)} /><span>{sc.title}<small>{sc.customerId ?? ""} {sc.facts.name ?? ""}</small></span></label>)}
                    <label><input type="radio" name="what" checked={!scenarioId} onChange={() => setScenarioId("")} /><span>Call freely<small>No rubric. Evaluated for timing, pronunciation, and audio quality only.</small></span></label>
                  </div></div>
                <div><div className="label" style={{ marginBottom: 6 }}>Optional: act as a persona (we show you their style; you do the voice)</div>
                  <div className={s.pick}>
                    {personas.data?.items.map((p) => <label key={p.id}><input type="radio" name="who" checked={personaId === p.id} onChange={() => setPersonaId(p.id)} /><span>{p.name}{p.age ? `, ${p.age}` : ""}<small>{p.temperament}, {p.speakingStyle.slice(0, 70)}</small></span></label>)}
                    <label><input type="radio" name="who" checked={!personaId} onChange={() => setPersonaId("")} /><span>Myself<small>No persona hints.</small></span></label>
                  </div>
                  <Field label="Line condition applied to your audio" style={{ marginTop: 12 }}><select className="in" value={line} onChange={(e) => setLine(e.target.value as typeof line)}><option value="none">none (clean)</option><option value="phone">8 kHz mu-law</option><option value="noisy">noisy mobile, 2% loss</option></select></Field>
                </div>
              </div>
              <div className="ph" style={{ borderTop: "1px solid var(--line-1)", borderBottom: 0, justifyContent: "flex-end", gap: 12 }}>
                {create.isError && <span style={{ color: "var(--fail)", fontSize: 12 }}>{(create.error as Error).message}</span>}
                <span className="faint">Profile <span className="mono">{profiles.data?.activeId ?? "…"}</span>{notReady.length ? <>, <HealthDot state={notReady[0].state}>{notReady.map((h) => h.name).join(", ")} {notReady[0].state === "down" ? "down" : "loading"}{notReady[0].elapsedS != null ? `, ${notReady[0].elapsedS} s` : ""}</HealthDot></> : ", bot models ready"}{health.data?.items.find((h) => h.role === "llm_judge")?.state === "loading" ? " (judge loading, evals queue)" : ""}</span>
                <Button variant="primary" size="lg" onClick={() => create.mutate()} disabled={create.isPending || notReady.length > 0 || !!mic.error}>{create.isPending ? "Connecting…" : <>Start call <Kbd>↵</Kbd></>}</Button>
              </div>
            </div>
          </div>
        )}
        {phase === "incall" && session && <InCall session={session} call={call} micLevel={mic.level} headphones={mic.headphonesLikely} onEnd={() => endM.mutate()} ending={endM.isPending} />}
        {phase === "evaluating" && endedCallId && <Evaluating callId={endedCallId} onAgain={() => { setPhase("setup"); setSession(null); setEndedCallId(null); }} onOpen={() => nav(`/calls/${endedCallId}`)} />}
      </div>
    </Shell>
  );
}

function Meter({ level, who, hot = 0.85 }: { level: number; who: "you" | "bot"; hot?: number }) {
  const n = 25; const on = Math.round(Math.min(1, level) * n);
  return <div className={`${s.meter} ${who === "bot" ? s.botMeter : ""}`} aria-label={`${who} level ${Math.round(level * 100)}%`} role="meter" aria-valuenow={Math.round(level * 100)} aria-valuemin={0} aria-valuemax={100}>{Array.from({ length: n }, (_, i) => <i key={i} className={`${i < on ? s.on : ""} ${i < on && i / n > hot ? s.hot : ""}`} />)}</div>;
}

function InCall({ session, call, micLevel, headphones, onEnd, ending }: { session: TalkSession; call: ReturnType<typeof useBotCall>; micLevel: number; headphones: boolean; onEnd: () => void; ending: boolean }) {
  const st = call.state;
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(t); }, []);
  const turnsRef = useRef<HTMLDivElement>(null);
  useEffect(() => { turnsRef.current?.scrollTo({ top: 1e9 }); }, [st.lines]);
  const elapsed = st.startedAt ? Math.max(0, now - st.startedAt) / 1000 : 0;
  const sheet = session.sheet;
  return (
    <div className={s.talk}>
      <div className="stack" style={{ minHeight: 0 }}>
        {sheet ? (
          <Panel title="Your sheet"><div className={s.sheet}>
            <div><b>You are</b> {sheet.youAre}</div>
            <div style={{ marginTop: 6 }}><b>The problem</b> {sheet.problem}</div>
            <div style={{ marginTop: 6 }}><b>Beats</b><ol>{sheet.beats.map((b) => <li key={b.id}>{b.instruction}</li>)}</ol></div>
            <div style={{ marginTop: 6 }}><b>Must not say</b> {sheet.mustNotSay.join(", ")}.</div>
          </div></Panel>
        ) : <Panel title="Free call"><div className={s.sheet}>No scenario. Say whatever you like; the call is evaluated for timing, pronunciation, and audio quality.</div></Panel>}
        <div className="panel flush" style={{ flex: 1, minHeight: 160, display: "flex", flexDirection: "column" }}><div className="ph"><h2>Events</h2></div>
          <div className={s.events} style={{ overflow: "auto", flex: 1 }}>
            <div><span className={s.ts}>00:00</span><span>{st.status === "connecting" ? "connecting…" : "call connected, bot speaking first"}</span></div>
            {st.events.map((e) => <div key={e.id}><span className={s.ts}>{fmtClock(e.atS)}</span><span className={e.kind === "barge" ? s.barge : e.kind === "warning" ? s.warning : ""}>{e.kind === "tool" ? "tool " : e.kind === "db" ? "db " : ""}{e.text}</span></div>)}
          </div></div>
      </div>
      <div className={`panel flush ${s.stage}`}>
        <div className="ph"><h2>Live call</h2><span className="mono faint">{session.callId}</span><ModeBadge mode="human" />{isMock() && <span className="tag">fixture mode</span>}<span className="grow" />
          {st.status === "connecting" && <HealthDot state="loading">connecting WebRTC</HealthDot>}
          {st.status === "live" && <HealthDot state="ok">bot responding{st.replyP50S ? `, p50 ${st.replyP50S} s` : ""}</HealthDot>}
          {st.status === "error" && <HealthDot state="down">{st.error}</HealthDot>}
          {st.status === "ended" && <HealthDot state="off">disconnected</HealthDot>}</div>
        {st.status === "error" && <Note kind="err" style={{ margin: 12 }}><b>Connection lost.</b> {st.error} The recording so far is kept. End the call to evaluate it, or call again.</Note>}
        <div className={s.strip}>
          <div className={`${s.spk} ${s.you} ${st.youSpeaking ? s.speaking : ""}`}><div className={s.who}><Icon name="talk" /> You</div><div className={`${s.state} ${st.youSpeaking ? "" : "muted"}`}>{st.muted ? "muted" : st.youSpeaking ? "speaking" : "quiet"}</div><Meter level={st.muted ? 0 : micLevel} who="you" /><div className="faint" style={{ fontSize: 11 }}>Echo cancellation on, gain control off.{headphones ? " Headphones detected." : " No headphones detected; expect some false interruptions on speakers."}</div></div>
          <div className={`${s.spk} ${s.bot} ${st.botSpeaking ? s.speaking : ""}`}><div className={s.who}>Bot <span className="faint" style={{ fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>Ada</span></div><div className={`${s.state} ${st.botSpeaking ? "" : "muted"}`}>{st.botSpeaking ? "speaking" : st.status === "live" ? "listening" : st.status}</div><Meter level={st.botSpeaking ? 0.55 : 0} who="bot" /><div className="faint" style={{ fontSize: 11 }}>{st.interruptions ? `Interrupted ${st.interruptions} time${st.interruptions === 1 ? "" : "s"}. Queued audio dropped.` : "Not interrupted yet. Talk over it to test barge-in."}</div></div>
        </div>
        <div className={s.turns} ref={turnsRef}>
          {st.lines.length === 0 && <div className="faint">Waiting for the bot's greeting…</div>}
          {st.lines.map((l) => <div key={l.id} className={`${s.lt} ${s[l.who]}`}><b>{l.who}</b><span className={l.final ? "" : "cursor"}>{l.text}</span></div>)}
        </div>
        <div className={s.endbar}><span className={s.clock}>{fmtClock(elapsed)}</span><span className="faint">Live transcript is from the bot's STT and may lag a second. The recording is what gets evaluated.</span><span className="grow" />
          <Button onClick={() => call.setMuted(!st.muted)} aria-pressed={st.muted}><Icon name={st.muted ? "mute" : "volume"} /> {st.muted ? "Unmute" : "Mute"} <Kbd>m</Kbd></Button>
          <Button variant="dangerSolid" size="lg" onClick={onEnd} disabled={ending}>End call <Kbd>Esc</Kbd><Kbd>Esc</Kbd></Button></div>
      </div>
    </div>
  );
}

function Evaluating({ callId, onAgain, onOpen }: { callId: string; onAgain: () => void; onOpen: () => void }) {
  const [evals, setEvals] = useState<EvaluatorRun[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [done, setDone] = useState<string | null>(null);
  const [closed, setClosed] = useState(false);
  const open = useCallback(() => endpoints.callSocket(callId), [callId]);
  useSocket<CallWsMessage>(open, (m) => {
    if (m.type === "evaluator") setEvals((e) => [...e.filter((x) => x.name !== m.evaluator.name), m.evaluator]);
    if (m.type === "finding") setFindings((f) => [...f, m.finding]);
    if (m.type === "eval") setDone(m.eval.verdict);
  }, { onClose: () => setClosed(true) });
  // Server-reported evaluators first (any new one appears automatically), then the expected set still to come.
  const order = orderedEvaluators(evals, findings, !done && !closed);
  const running = order.find((n) => !evals.some((e) => e.name === n));
  return (
    <div className="panel" style={{ maxWidth: 720, margin: "0 auto" }}>
      <div className="ph"><h2>Call ended, evaluating</h2><span className="mono faint">{callId}</span><ModeBadge mode="human" /><span className="grow" />{done ? <VerdictBadge v={done as "pass"} /> : closed ? <span className="faint">connection closed</span> : <HealthDot state="loading">running</HealthDot>}</div>
      <div className="pb">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}><tbody>
          <tr><td style={{ width: 90, padding: "6px 0" }}><span className="v pass">done</span></td><td>Recording saved<div className="faint">stereo, you left, bot right</div></td></tr>
          {order.map((n) => { const e = evals.find((x) => x.name === n); const fs = findings.filter((f) => f.evaluator === n); return (
            <tr key={n}><td style={{ padding: "6px 0" }}>{e ? (e.status === "skipped" ? <span className="v queued">skipped</span> : e.status === "error" ? <VerdictBadge v="failed" /> : <span className="v pass">done</span>) : n === running && !done && !closed ? <VerdictBadge v="running" /> : <VerdictBadge v="queued" />}</td>
              <td>{evalName(n)}{isAdvisoryEvaluator(n, findings) ? " (advisory)" : ""}<div className="faint">{e?.status === "skipped" ? e.reason : fs.length ? `${fs.length} finding${fs.length === 1 ? "" : "s"}: ${fs.map((f) => f.title).join("; ")}` : e ? "no findings" : n === "transcript_judge" && n === running ? "waiting on the judge LLM" : ""}</div></td></tr>); })}
        </tbody></table>
        <div className="btn-row" style={{ marginTop: 24 }}><Button variant="primary" onClick={onOpen}>{done ? "Open review" : "Open review as results arrive"}</Button><Button onClick={onAgain}>Call again</Button><span className="faint">You can leave. The call appears in the review queue when evals finish.</span></div>
      </div>
    </div>
  );
}
