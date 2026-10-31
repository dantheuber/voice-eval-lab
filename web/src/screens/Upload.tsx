import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Field, HealthDot, Icon, Note, Panel } from "@/components/ui";
import { useScenarios } from "@/api/queries";
import { endpoints } from "@/api/client";
import { fmtDuration } from "@/lib/format";
import s from "./screens.module.css";

export function UploadScreen() {
  const scenarios = useScenarios();
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [scenarioId, setScenarioId] = useState("");
  const [botVersion, setBotVersion] = useState("");
  const [intended, setIntended] = useState("");
  const [toolLog, setToolLog] = useState("");
  const [channelMap, setChannelMap] = useState<"caller_left" | "caller_right" | "mono">("caller_left");
  const inputRef = useRef<HTMLInputElement>(null);
  const accept = /\.(wav|flac|mp3|ogg)$/i;

  const upload = useMutation({
    mutationFn: async () => {
      const form = new FormData();
      form.set("file", file!);
      if (scenarioId) form.set("scenarioId", scenarioId);
      if (botVersion) form.set("botVersion", botVersion);
      if (intended.trim()) form.set("intendedTranscript", intended);
      if (toolLog.trim()) form.set("toolLog", toolLog);
      return endpoints.upload(form);
    },
  });
  const evaluate = useMutation({
    mutationFn: async () => { const r = upload.data!; if (r.probe.channels === 1 || r.probe.channelMap === "unknown") await endpoints.uploadChannelMap(r.callId, channelMap); await endpoints.evaluateUpload(r.callId); return r.callId; },
    onSuccess: (callId) => nav(`/calls/${callId}`),
  });
  const pick = (f: File | null) => { if (f && !accept.test(f.name)) { alert("Unsupported file. Use WAV, FLAC, MP3, or OGG."); return; } setFile(f); upload.reset(); };

  const ev = upload.data?.evidence ?? { audio: !!file, transcript: !!file, intendedText: !!intended.trim(), toolLog: !!toolLog.trim(), dbSnapshots: false, scenario: !!scenarioId };
  const rows: [boolean, string, string][] = [
    [ev.audio, "Audio, both channels", "timing, overlap, audio quality, listening"],
    [ev.transcript, "Transcript from our STT", "transcript judge, general"],
    [ev.scenario, "Scenario and rubric", ev.scenario ? "rubric scored by the judge" : "pick a scenario to enable"],
    [ev.intendedText, "Intended bot text", ev.intendedText ? "pronunciation vs intended text" : "pronunciation reduced to ASR vs ASR"],
    [ev.toolLog, "Tool log", ev.toolLog ? "tool checks" : "tool checks skipped"],
    [ev.dbSnapshots, "Database before and after", "state assertions skipped"],
  ];
  const reduced = !ev.toolLog || !ev.dbSnapshots || !ev.scenario;

  return (
    <Shell crumbs={["Upload recording"]}>
      <div className="page">
        <div className="page-head"><div><h1>Upload a recording</h1><div className="page-sub">Evaluate a call recorded elsewhere. Less evidence means fewer checks. The review screen says what could not be checked.</div></div></div>
        <div className={s.builder}>
          <div className="stack">
            <Panel>
              <div className={`${s.drop} ${over ? s.over : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0] ?? null); }}>
                <b>Drop a WAV, FLAC, MP3, or OGG here</b>Stereo with caller left and bot right gives the best results. Mono is accepted and split by energy, which is a guess, so reply and overlap timing are skipped.
                <div className="btn-row" style={{ justifyContent: "center", marginTop: 12 }}><Button onClick={() => inputRef.current?.click()}>Choose file</Button><input ref={inputRef} type="file" accept=".wav,.flac,.mp3,.ogg,audio/*" hidden onChange={(e) => pick(e.target.files?.[0] ?? null)} /></div>
              </div>
              {file && (
                <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 12, padding: "8px 12px", border: "1px solid var(--line-1)", borderRadius: 6, fontSize: 12, flexWrap: "wrap" }}>
                  <span className="mono">{file.name}</span><span className="faint">{(file.size / 1e6).toFixed(1)} MB</span>
                  {upload.data && <span className="faint">{fmtDuration(upload.data.probe.durationS)} · {upload.data.probe.channels === 2 ? "stereo" : "mono"} · {upload.data.probe.sampleRate / 1000} kHz</span>}
                  <span className="grow" />
                  {upload.isPending ? <HealthDot state="loading">probing</HealthDot> : upload.data ? <HealthDot state={upload.data.probe.channelMap === "unknown" || upload.data.probe.channels === 1 ? "loading" : "ok"}>{upload.data.probe.channelMap === "caller_left" ? "channels detected: caller L, bot R" : upload.data.probe.channelMap === "caller_right" ? "channels detected: caller R, bot L" : upload.data.probe.channels === 1 ? "mono, pick the bot speaker below" : "channel map unknown"}</HealthDot> : <Button size="sm" onClick={() => upload.mutate()}>Probe file</Button>}
                  {upload.isError && <span style={{ color: "var(--fail)" }}>{(upload.error as Error).message}</span>}
                </div>
              )}
              {upload.data && (upload.data.probe.channels === 1 || upload.data.probe.channelMap === "unknown") && (
                <Field label={upload.data.probe.channels === 1 ? "Who speaks first?" : "Which channel is the caller?"} style={{ marginTop: 12 }} help="Listen to the first 10 seconds if you're not sure.">
                  <div className="btn-row">
                    {upload.data.probe.channels === 1
                      ? <select className="in" aria-label="Who speaks first" value={channelMap} onChange={(e) => setChannelMap(e.target.value as typeof channelMap)} style={{ width: "auto" }}><option value="caller_left">the bot, then the caller</option><option value="caller_right">the caller, then the bot</option></select>
                      : <select className="in" aria-label="Which channel is the caller" value={channelMap} onChange={(e) => setChannelMap(e.target.value as typeof channelMap)} style={{ width: "auto" }}><option value="caller_left">caller left, bot right</option><option value="caller_right">caller right, bot left</option><option value="mono">one mixed channel, split by energy</option></select>}
                    <audio controls preload="none" src={endpoints.mediaUrl(`/api/uploads/${upload.data.callId}/preview?startS=0&durationS=10`) ?? undefined} style={{ height: 32 }} />
                  </div>
                </Field>
              )}
            </Panel>
            <Panel title="What we know about this call" meta="optional, each field unlocks more checks">
              <div className="stack">
                <div className="form-row">
                  <Field label="Scenario" help="With a scenario the judge can score the rubric."><select className="in" value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}><option value="">unknown (judge from transcript only)</option>{scenarios.data?.items.map((sc) => <option key={sc.id} value={sc.id}>{sc.title}</option>)}</select></Field>
                  <Field label="Bot version or profile"><input className="in" value={botVersion} onChange={(e) => setBotVersion(e.target.value)} placeholder="e.g. prod 2026-10-28" /></Field>
                </div>
                <Field label="Bot transcript with intended text (optional)"><textarea className="in mono" value={intended} onChange={(e) => setIntended(e.target.value)} placeholder="One bot turn per line, or JSON [{ text, startS }]. Without this, pronunciation can only compare two ASR passes." /></Field>
                <Field label="Tool log (optional)"><textarea className="in mono" value={toolLog} onChange={(e) => setToolLog(e.target.value)} placeholder="One JSON tool call per line. Without this, tool and database checks are skipped." /></Field>
              </div>
            </Panel>
          </div>
          <div className="stack sticky-col">
            <Panel title="Evidence available">
              <div className={s.evidence}>{rows.map(([ok, what, why]) => <div key={what}><span className={ok ? s.ok : s.no}><Icon name={ok ? "check" : "x"} /></span><span>{what}</span><span className={s.r}>{why}</span></div>)}</div>
              {reduced && <Note kind="warn" style={{ marginTop: 12 }}>With this evidence the call cannot be marked <b>pass</b> or <b>fail</b> automatically. It will land in <b>needs review</b> with findings from the checks that could run.</Note>}
              <div className="btn-row" style={{ marginTop: 12 }}>
                {!upload.data ? <Button variant="primary" size="lg" disabled={!file || upload.isPending} onClick={() => upload.mutate()}>{upload.isPending ? "Uploading…" : "Upload and probe"}</Button>
                  : <Button variant="primary" size="lg" disabled={evaluate.isPending} onClick={() => evaluate.mutate()}>{evaluate.isPending ? "Starting…" : "Evaluate"}</Button>}
                <span className="faint">{upload.data ? `about ${Math.round(upload.data.probe.durationS / 8)} s for ${fmtDuration(upload.data.probe.durationS)} of audio` : "about 40 s for 5 min of audio"}</span>
              </div>
              {evaluate.isError && <Note kind="err" style={{ marginTop: 8 }}>{(evaluate.error as Error).message}</Note>}
            </Panel>
          </div>
        </div>
      </div>
    </Shell>
  );
}
