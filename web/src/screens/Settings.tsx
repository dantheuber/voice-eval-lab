import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Field, HealthDot, Note, Panel, QueryState, Skeleton, Untested } from "@/components/ui";
import { keys, useAws, useHealth, useInvalidate, useProfiles, useRoles, useSystem, useThresholds } from "@/api/queries";
import { endpoints } from "@/api/client";
import { ApiError } from "@/api/transport";
import { toast } from "@/store/toast";
import { fmtWhen } from "@/lib/format";
import type { Thresholds } from "@/api/types";
import s from "./screens.module.css";

/* A swap changes nothing when it fails. This stays up until the next try, since what's missing takes a while to fix. */
function SwapError({ error }: { error: unknown }) {
  const failure = error instanceof ApiError && error.code === "not_installed" ? (error.detail as { missing: string[]; setup: string | null }) : null;
  if (!failure) return <Note kind="err"><b>Not switched.</b> {(error as Error).message}</Note>;
  return (
    <Note kind="err">
      <b>This isn't installed, so nothing changed.</b> Missing:
      <span style={{ display: "block", maxHeight: 120, overflow: "auto", margin: "4px 0" }}>{failure.missing.map((path) => <span key={path} className="mono" style={{ display: "block" }}>{path}</span>)}</span>
      {failure.setup ? <>Run <span className="mono">{failure.setup}</span>, then pick it again.</> : "Install it, or fix the path in the profile's YAML."}
    </Note>
  );
}

export function SettingsScreen() {
  const profiles = useProfiles();
  const roles = useRoles();
  const health = useHealth();
  const system = useSystem();
  const thresholds = useThresholds();
  const aws = useAws();
  const invalidate = useInvalidate();
  const [swapError, setSwapError] = useState<unknown>(null);
  const setProfile = useMutation({ mutationFn: (id: string) => endpoints.setActiveProfile(id), onMutate: () => setSwapError(null), onError: (e) => setSwapError(e), onSuccess: () => { invalidate(keys.profiles, keys.roles, keys.health); toast("Profile switched. Its servers are ready."); } });
  const setRole = useMutation({ mutationFn: ({ role, adapter }: { role: string; adapter: string }) => endpoints.setRole(role, adapter), onMutate: () => setSwapError(null), onSuccess: (r) => { invalidate(keys.roles, keys.health, keys.profiles); toast(`${r.label} now uses ${r.options.find((o) => o.id === r.adapter)?.label ?? r.adapter}`); }, onError: (e) => setSwapError(e) });
  const restart = useMutation({ mutationFn: (role: string) => endpoints.restartServer(role), onSuccess: () => { invalidate(keys.health, keys.roles); toast("Restarting"); } });
  const restartAll = useMutation({ mutationFn: () => endpoints.restartAll(), onSuccess: () => { invalidate(keys.health, keys.roles); toast("Restarting all servers"); } });
  const bench = useMutation({ mutationFn: () => endpoints.benchmark(), onSuccess: () => toast("Benchmark started. Result lands on the profile when done.") });
  const awsTest = useMutation({ mutationFn: () => endpoints.awsTest() });
  const [th, setTh] = useState<Thresholds | null>(null);
  useEffect(() => { if (thresholds.data && !th) setTh(thresholds.data); }, [thresholds.data, th]);
  const saveTh = useMutation({ mutationFn: () => endpoints.setThresholds(th!), onSuccess: () => { invalidate(keys.thresholds); toast("Thresholds saved"); } });
  const active = profiles.data?.items.find((p) => p.id === profiles.data?.activeId);
  const loading = health.data?.items.filter((h) => h.state === "loading") ?? [];

  return (
    <Shell crumbs={["Models and settings"]}>
      <div className="page">
        <div className="page-head"><div><h1>Models and settings</h1><div className="page-sub">Profiles are YAML files in <span className="mono">config/</span>. Changing a role here writes the active profile and restarts only that server.</div></div>
          <div className="actions"><Button onClick={() => bench.mutate()} disabled={bench.isPending}>Run benchmark</Button><Button onClick={() => { if (confirm("Restart all model servers? Running calls will fail.")) restartAll.mutate(); }}>Restart all servers</Button></div></div>
        <div className="cols-main-side">
          <div className="stack">
            <Panel flush title="Active profile" meta={active ? <>{active.configPath} · {active.liveCapable ? "live mode ok" : "turn-based only"}{active.benchmark ? `, models alone answer in ${active.benchmark.botReplyP50S} s, before voice detection and turn waits, benchmark ${fmtWhen(active.benchmark.at)}` : ", no benchmark yet"}</> : ""}
              actions={<>{setProfile.isPending && <HealthDot state="loading">starting {setProfile.variables}</HealthDot>}<select className="in" style={{ width: "auto", height: 26 }} value={profiles.data?.activeId ?? ""} disabled={setProfile.isPending} onChange={(e) => setProfile.mutate(e.target.value)} aria-label="Active profile">{profiles.data?.items.map((p) => <option key={p.id} value={p.id}>{p.label}{p.untested ? " (untested)" : ""}</option>)}</select></>}>
              {swapError != null && <div style={{ padding: 12 }}><SwapError error={swapError} /></div>}
              <div style={{ overflow: "auto" }}>
                <QueryState q={roles} what="roles" skeleton={<div style={{ padding: 12 }}><Skeleton rows={8} /></div>}>
                  {(d) => (
                    <table className={`t ${s.roles}`}>
                      <thead><tr><th>Role</th><th>Interface</th><th>Adapter</th><th>Status</th><th>Licence</th><th></th></tr></thead>
                      <tbody>{d.items.map((r) => { const opt = r.options.find((o) => o.id === r.adapter); return (
                        <tr key={r.role}>
                          <td><b>{r.label}</b></td><td className="mono faint">{r.interface}</td>
                          <td><span style={{ display: "inline-flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}><select className="in" aria-label={`${r.label} adapter`} value={r.adapter} disabled={setRole.isPending || setProfile.isPending} onChange={(e) => setRole.mutate({ role: r.role, adapter: e.target.value })}>{r.options.map((o) => <option key={o.id} value={o.id}>{o.label}{o.untested ? " (untested)" : ""}</option>)}</select>{opt?.untested && <Untested />}</span></td>
                          <td>{r.health.state === "off" ? <span className="faint">off</span> : <HealthDot state={r.health.state}>{r.health.detail}{r.health.state === "loading" && r.health.elapsedS != null ? `, ${r.health.elapsedS} s` : ""}</HealthDot>}</td>
                          <td className="faint" style={{ fontSize: 11 }}>{opt?.license}</td>
                          <td>{r.health.state !== "off" && <Button size="sm" variant="ghost" onClick={() => restart.mutate(r.role)}>restart</Button>}</td>
                        </tr>); })}</tbody>
                    </table>
                  )}
                </QueryState>
              </div>
            </Panel>
            <Panel title="Swap a provider">
              <Note kind="warn"><b>Bedrock adapters are untested against live AWS.</b> There is no Bedrock access on this machine. Adapters were unit tested against mocks only. Choosing one shows this label on every call made with it, and on every result in Compare.</Note>
              <div className="form-row" style={{ marginTop: 12 }}>
                <Field label="AWS region" help="set AWS_PROFILE or AWS keys in the environment before starting"><input className="in" value={aws.data?.region ?? ""} disabled /></Field>
                <Field label="Credentials" help={aws.data?.problem ?? undefined}><HealthDot state={aws.data?.credentials === "found" ? "ok" : "down"}>{aws.data?.credentials === "found" ? "found" : "not found"}</HealthDot></Field>
                <Field label="Connection" help={aws.data?.credentials !== "found" ? "Needs credentials first." : "Makes one request to Bedrock."}><div><Button disabled={aws.data?.credentials !== "found" || awsTest.isPending} onClick={() => awsTest.mutate()}>Test connection</Button></div>{awsTest.data && <span className="help" style={{ color: awsTest.data.ok ? "var(--pass)" : "var(--fail)" }}>{awsTest.data.message}</span>}</Field>
              </div>
            </Panel>
            <Panel title="Evaluation thresholds" actions={<Button size="sm" variant="primary" disabled={!th || saveTh.isPending} onClick={() => saveTh.mutate()}>Save</Button>}>
              {th ? (
                <div className="form-row">
                  <Field label="Dead air (s)"><input className="in" type="number" step={0.1} value={th.deadAirS} onChange={(e) => setTh({ ...th, deadAirS: +e.target.value })} /></Field>
                  <Field label="Barge-in stop target (s)"><input className="in" type="number" step={0.1} value={th.bargeStopTargetS} onChange={(e) => setTh({ ...th, bargeStopTargetS: +e.target.value })} /></Field>
                  <Field label="Reply latency warn (s)"><input className="in" type="number" step={0.1} value={th.replyLatencyWarnS} onChange={(e) => setTh({ ...th, replyLatencyWarnS: +e.target.value })} /></Field>
                  <Field label="MOS baseline"><input className="in" type="number" step={0.1} value={th.mosBaseline} onChange={(e) => setTh({ ...th, mosBaseline: +e.target.value })} /></Field>
                  <Field label="Audio listening" help="Fixed. Agreement with humans is under 80%."><select className="in" value="advisory" disabled><option value="advisory">advisory only (cannot fail a call)</option></select></Field>
                </div>
              ) : <Skeleton rows={2} />}
            </Panel>
          </div>
          <div className="stack">
            <Panel flush title="Model servers" actions={loading.length ? <HealthDot state="loading">{loading.length} loading</HealthDot> : <HealthDot state="ok">all ready</HealthDot>}>
              <QueryState q={health} what="model health" skeleton={<div style={{ padding: 12 }}><Skeleton rows={6} /></div>}>
                {(d) => <>
                  {d.items.map((h) => <div key={h.role} className={s.serverRow}><HealthDot state={h.state} /><div style={{ flex: 1, minWidth: 0 }}><div>{h.name}</div><div className={s.m}>{h.model}</div></div><span className={h.state === "down" ? "faint" : "faint nowrap"}>{h.detail}{h.state === "loading" && h.elapsedS != null ? `, ${h.elapsedS} s` : ""}</span>{h.state === "down" && <Button size="sm" variant="ghost" onClick={() => restart.mutate(h.role)}>restart</Button>}</div>)}
                  {loading.map((h) => <div key={h.role} style={{ padding: "8px 12px" }}><Note kind="warn">{h.name} loading{h.elapsedS != null ? `, ${h.elapsedS} s` : ""}. {h.usualS != null ? `Usually under ${h.usualS} s on this GPU.` : ""} Evals queue until it reports ready; calls are not blocked.</Note></div>)}
                </>}
              </QueryState>
            </Panel>
            <Panel flush title="GPU">
              {system.data ? <dl className="kv"><dt>device</dt><dd>{system.data.gpu.device}</dd><dt>memory in use</dt><dd>{system.data.gpu.memoryUsedGb} of {system.data.gpu.memoryTotalGb} GB</dd><dt>servers</dt><dd>{system.data.servers.map((x) => `${x.kind}:${x.port}`).join(", ")}</dd><dt>Ollama</dt><dd className="plain">not managed here</dd></dl> : <div style={{ padding: 12 }}><Skeleton rows={3} /></div>}
            </Panel>
            <Panel flush title="Access">
              {system.data && <dl className="kv"><dt>local</dt><dd>{system.data.urls.local}</dd><dt>LAN</dt><dd>{system.data.urls.lan}</dd><dt>cert</dt><dd>{system.data.cert}<span className="plain faint">, accept once per browser</span></dd></dl>}
            </Panel>
          </div>
        </div>
      </div>
    </Shell>
  );
}
