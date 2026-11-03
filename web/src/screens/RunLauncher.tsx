import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Field, HealthDot, Kbd, ModeBadge, Note, Panel, PassFrac, Seg, Skeleton } from "@/components/ui";
import { useHealth, useInvalidate, useMatrix, usePersonas, useProfiles, useScenarios, keys } from "@/api/queries";
import { endpoints } from "@/api/client";
import { useHotkeys } from "@/hooks/useHotkeys";
import { interruptionScale } from "@/lib/format";
import { passPct, valid } from "@/lib/stats";
import type { CallMode, LineConditions, RunRequest } from "@/api/types";
import s from "./screens.module.css";

const LINES: Record<string, LineConditions> = {
  clean: { codec: "none", noise: "none", snrDb: 30, packetLossPct: 0, oneWayLatencyMs: 40 },
  phone: { codec: "mulaw_8k", noise: "none", snrDb: 22, packetLossPct: 0, oneWayLatencyMs: 80 },
  noisy: { codec: "mulaw_8k", noise: "car", snrDb: 12, packetLossPct: 2, oneWayLatencyMs: 120 },
};

export function RunLauncherScreen() {
  const scenarios = useScenarios();
  const personas = usePersonas();
  const profiles = useProfiles();
  const health = useHealth();
  const nav = useNavigate();
  const invalidate = useInvalidate();
  const [profile, setProfile] = useState<string>("");
  const prof = profiles.data?.items.find((p) => p.id === (profile || profiles.data?.activeId)) ?? null;
  const matrix = useMatrix(prof?.id);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [reps, setReps] = useState(3);
  const [mode, setMode] = useState<"auto" | "live_bots" | "turn_based">("auto");
  const [line, setLine] = useState<"persona" | "clean" | "phone" | "noisy">("persona");
  const [wanted, setConcurrency] = useState(2);
  const [label, setLabel] = useState("");
  const key = (sc: string, pe: string) => `${sc}:${pe}`;
  const toggle = (k: string, on: boolean) => setSel((prev) => { const n = new Set(prev); if (on) n.add(k); else n.delete(k); return n; });
  const cells = useMemo(() => new Map((matrix.data?.cells ?? []).map((c) => [key(c.scenarioId, c.personaId), c.counts])), [matrix.data]);
  const pairs = [...sel].map((k) => { const [scenarioId, personaId] = k.split(":"); return { scenarioId, personaId }; });
  const concurrency = Math.min(wanted, prof?.maxCalls ?? wanted);
  const calls = pairs.length * reps;
  const effectiveMode: CallMode = mode === "auto" ? (prof?.liveCapable ? "live_bots" : "turn_based") : mode;
  const perCall = effectiveMode === "live_bots" ? 210 : 420;
  const est = Math.round((calls * perCall) / 60 / Math.max(1, concurrency));
  const judge = health.data?.items.find((h) => h.role === "llm_judge");
  const launch = useMutation({
    mutationFn: () => { const body: RunRequest = { pairs, reps, profile: prof!.id, mode, line: line === "persona" ? "persona" : LINES[line], concurrency, label: label || null }; return endpoints.createRun(body); },
    onSuccess: (r) => { invalidate(keys.runs, keys.overview()); nav(`/runs/${r.run.id}`); },
  });
  useHotkeys({ "Ctrl+Enter": () => { if (calls && !launch.isPending) launch.mutate(); } });
  const loading = scenarios.isPending || personas.isPending || profiles.isPending;
  const scs = scenarios.data?.items ?? [];
  const pes = personas.data?.items ?? [];

  return (
    <Shell crumbs={["Run launcher", "New run"]} keys={[["Launch", "Ctrl Enter"]]}>
      <div className="page">
        <div className="page-head"><div><h1>New run</h1><div className="page-sub">Pick scenario and persona pairs, set repetitions, launch. Each call gets a fresh copy of the Fernhill database.</div></div></div>
        {loading ? <Skeleton rows={8} /> : (
          <div className={s.launch}>
            <Panel flush title="Scenarios × personas" meta={`${pairs.length} pairs selected. Cells show the last 7 days' pass rate on ${prof?.id ?? "the active profile"}.`} actions={<><Button size="sm" variant="ghost" onClick={() => setSel(new Set(scs.flatMap((sc) => pes.map((pe) => key(sc.id, pe.id)))))}>Select all</Button><Button size="sm" variant="ghost" onClick={() => setSel(new Set())}>Clear</Button></>}>
              <div style={{ overflow: "auto" }}>
                {scs.length === 0 || pes.length === 0 ? <div className="empty"><b>Nothing to run</b>You need at least one scenario and one persona.</div> : (
                  <table className={s.matrix}>
                    <thead><tr><th></th>{pes.map((pe) => <th key={pe.id} className={s.p}><label className="check" style={{ flexDirection: "column", gap: 2 }} title={`${pe.name}: ${pe.temperament}, interrupts ${interruptionScale(pe.interruptionRate)}/5`}><input type="checkbox" checked={scs.every((sc) => sel.has(key(sc.id, pe.id)))} onChange={(e) => scs.forEach((sc) => toggle(key(sc.id, pe.id), e.target.checked))} aria-label={`all scenarios for ${pe.name}`} /> {pe.name.split(" ").slice(0, 2).join(" ")}</label><small>{pe.temperament}</small></th>)}<th className={s.p} title="select every persona on a row"><small>whole row</small></th></tr></thead>
                    <tbody>{scs.map((sc) => (
                      <tr key={sc.id}>
                        <td className={s.s}>{sc.title}<small>{sc.customerId ?? ""}</small></td>
                        {pes.map((pe) => { const c = cells.get(key(sc.id, pe.id)); const p = c ? passPct(c) : null; return (
                          <td key={pe.id}><label className="check" style={{ justifyContent: "center", flexDirection: "column", gap: 1 }}><input type="checkbox" checked={sel.has(key(sc.id, pe.id))} onChange={(e) => toggle(key(sc.id, pe.id), e.target.checked)} aria-label={`${sc.title} with ${pe.name}`} /><span className={`${s.hist} ${p != null && p < 70 ? s.low : ""} ${c && valid(c) ? "" : s.none}`} title={c && valid(c) ? undefined : c ? `${c.invalid} invalid call${c.invalid === 1 ? "" : "s"} only in the last 7 days` : "no calls for this pair in the last 7 days"}>{c && valid(c) ? <PassFrac pass={c.pass} n={valid(c)} bold={false} compact /> : "–"}</span></label></td>); })}
                        <td><input type="checkbox" checked={pes.every((pe) => sel.has(key(sc.id, pe.id)))} onChange={(e) => pes.forEach((pe) => toggle(key(sc.id, pe.id), e.target.checked))} aria-label={`all personas for ${sc.title}`} /></td>
                      </tr>))}</tbody>
                  </table>
                )}
              </div>
            </Panel>
            <div className="stack">
              <Panel title="Settings">
                <div className="stack">
                  <Field label="Repetitions per pair" help="Both sides are stochastic. 3 or more gives a pass rate you can read; 1 only tells you whether the pair can pass at all."><Seg value={String(reps)} options={[1, 2, 3, 5, 10].map((n) => ({ value: String(n), label: n }))} onChange={(v) => setReps(+v)} /></Field>
                  <Field label="Model profile" help={prof?.description}><select className="in" value={prof?.id ?? ""} onChange={(e) => setProfile(e.target.value)}>{profiles.data?.items.map((p) => <option key={p.id} value={p.id} disabled={p.health === "down"}>{p.label}{p.health === "loading" ? " (loading)" : p.health === "down" ? " (unavailable)" : ""}{p.untested ? " (untested)" : ""}</option>)}</select></Field>
                  <Field label="Call mode" help={<>{effectiveMode === "live_bots" ? <>Live: real time, full duplex, timing metrics measured.{prof?.benchmark ? ` Benchmark says this profile keeps up (bot reply p50 ${prof.benchmark.botReplyP50S} s).` : ""}</> : <>Turn-based: slower than real time, each side waits. Timing metrics are not measured. Calls will carry a dashed <b>turn-based</b> badge.</>}{mode === "auto" ? " Chosen from the last benchmark." : ""}</>}>
                    <Seg value={mode} options={[{ value: "auto", label: "auto" }, { value: "live_bots", label: "live" }, { value: "turn_based", label: "turn-based" }]} onChange={setMode} /></Field>
                  <Field label="Line conditions" help="Per persona uses each persona's own line. Overriding applies one condition to every call."><Seg value={line} options={[{ value: "persona", label: "per persona" }, { value: "clean", label: "clean" }, { value: "phone", label: "8 kHz" }, { value: "noisy", label: "noisy mobile" }]} onChange={setLine} /></Field>
                  <Field label="Concurrency" help={prof?.maxCalls ? `This profile has room for ${prof.maxCalls} call at a time.` : effectiveMode === "live_bots" ? "2 concurrent live calls fit the GPU." : "Turn-based can go to 4."}><input className="in" type="number" min={1} max={prof?.maxCalls ?? 4} disabled={prof?.maxCalls === 1} value={concurrency} onChange={(e) => setConcurrency(Math.max(1, Math.min(4, +e.target.value)))} style={{ width: 60 }} /></Field>
                  <Field label="Label (optional)"><input className="in" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. after prompt change" /></Field>
                </div>
              </Panel>
              <Panel title="Summary">
                <dl className={s.summaryL}><dt>calls</dt><dd><b>{calls}</b> ({pairs.length} pairs × {reps})</dd><dt>mode</dt><dd><ModeBadge mode={effectiveMode} /></dd><dt>estimate</dt><dd>about {est} min at {concurrency} concurrent{effectiveMode === "turn_based" ? ", wall clock, not call time" : ""}</dd><dt>evals</dt><dd>run after each call, judge LLM {judge?.state === "loading" ? <HealthDot state="loading">loading</HealthDot> : judge?.state === "down" ? <HealthDot state="down">down</HealthDot> : "ready"}</dd></dl>
                <div className="btn-row" style={{ marginTop: 12 }}><Button variant="primary" size="lg" disabled={!calls || launch.isPending || !prof} onClick={() => launch.mutate()}>Launch {calls} calls <Kbd>Ctrl</Kbd><Kbd>↵</Kbd></Button></div>
                {calls === 0 && <Note kind="warn" style={{ marginTop: 8 }}>Select at least one scenario and persona pair.</Note>}
                {prof?.untested && <Note kind="warn" style={{ marginTop: 8 }}>This profile uses adapters untested against live AWS. Every call will carry that label.</Note>}
                {launch.isError && <Note kind="err" style={{ marginTop: 8 }}>{(launch.error as Error).message}</Note>}
              </Panel>
            </div>
          </div>
        )}
      </div>
    </Shell>
  );
}
