import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Empty, Field, HealthDot, Icon, Kbd, ModeBadge, Note, Panel, PassRate, QueryState, Scale, Skeleton, VerdictBadge } from "@/components/ui";
import { keys, useInvalidate, usePersona, usePersonas, useVoices } from "@/api/queries";
import { endpoints } from "@/api/client";
import { useHotkeys } from "@/hooks/useHotkeys";
import { toast } from "@/store/toast";
import { interruptionRate, interruptionScale, usageLine } from "@/lib/format";
import { ApiError } from "@/api/transport";
import type { LineConditions, Persona, PersonaInput } from "@/api/types";
import s from "./screens.module.css";

export function PersonasScreen() {
  const q = usePersonas();
  return (
    <Shell crumbs={["Personas"]} actions={<Link to="/personas/new" className="btn sm primary">New persona</Link>}>
      <div className="page">
        <div className="page-head"><div><h1>Personas</h1><div className="page-sub">Who the caller bot pretends to be. Each persona also sets the line conditions its audio goes through.</div></div></div>
        <Panel flush>
          <QueryState q={q} what="personas" skeleton={<div style={{ padding: 12 }}><Skeleton rows={6} /></div>}>
            {(d) => d.items.length === 0 ? <Empty title="No personas yet"><Link to="/personas/new">Create one</Link> or seed from <span className="mono">seeds/personas</span>.</Empty> : (
              <div className={s.list} role="table" aria-label="Personas">
                <div className={s.listHead} role="row" aria-hidden="true"><span>Persona</span><span className={s.mid}>Temperament</span><span className={s.num} title="Pass rate of valid calls with the 95% interval">Pass rate</span><span className={s.id}>Id</span></div>
                {d.items.map((p) => (
                  <Link key={p.id} to={`/personas/${p.id}`} className={s.listRow} role="row" title={p.id}>
                    <span><b>{p.name}</b><small className={s.desc} title={p.summary || p.speakingStyle}>{p.summary || p.speakingStyle}</small></span>
                    <span className={s.mid}>{p.temperament}{p.age ? `, ${p.age}` : ""}<small>interrupts {interruptionScale(p.interruptionRate)}/5, patience {p.patience}/5</small></span>
                    <span className={s.num}>{p.usage.calls ? <PassRate counts={p.usage.counts} /> : <span className="faint">no calls yet</span>}<small>{usageLine(p.usage)}</small></span>
                    <span className={s.id}>{p.id}</span>
                  </Link>
                ))}
              </div>
            )}
          </QueryState>
        </Panel>
      </div>
    </Shell>
  );
}

/* `default_female_1` and the like stand for a matching voice on whichever TTS the profile uses, so a persona works on all of them. */
const voiceLabel = (id: string) => (id.startsWith("default_") ? `${id.slice(8).replace("_", " ")}, the profile's own voice` : id);

const blank = (): PersonaInput => ({ name: "", summary: "", age: null, temperament: "neutral", speakingStyle: "", patience: 3, verbosity: 3, techSavviness: 3, interruptionRate: 0, voice: { provider: null, voiceId: "default_female_1", style: null }, line: { codec: "mulaw_8k", noise: "none", snrDb: 20, packetLossPct: 0, oneWayLatencyMs: 80 } });

/* Editor state is keyed by resource id, so jumping from one persona to another never carries the previous form over. */
export function PersonaEditorScreen() {
  const { id } = useParams();
  return <PersonaEditor key={id ?? "new"} id={id} />;
}

function PersonaEditor({ id }: { id: string | undefined }) {
  const isNew = !id;
  const q = usePersona(id);
  const voices = useVoices();
  const nav = useNavigate();
  const invalidate = useInvalidate();
  const [form, setForm] = useState<PersonaInput | null>(isNew ? blank() : null);
  const [dirty, setDirty] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  useEffect(() => { if (q.data && !form) setForm(q.data); }, [q.data, form]);
  const up = <K extends keyof PersonaInput>(k: K, v: PersonaInput[K]) => { setForm((f) => (f ? { ...f, [k]: v } : f)); setDirty(true); };
  const upLine = <K extends keyof LineConditions>(k: K, v: LineConditions[K]) => { setForm((f) => (f ? { ...f, line: { ...f.line, [k]: v } } : f)); setDirty(true); };

  const save = useMutation({
    mutationFn: async () => { if (!form) throw new Error("no form"); if (!form.name.trim()) throw new ApiError(400, { code: "invalid", message: "Name is required." }, "invalid"); return isNew ? endpoints.createPersona(form) : endpoints.updatePersona(id!, { ...(q.data as Persona), ...form, id: id! }); },
    onSuccess: (p) => { invalidate(keys.personas, keys.persona(p.id)); setDirty(false); setSavedAt(Date.now()); toast("Saved"); if (isNew) nav(`/personas/${p.id}`, { replace: true }); },
  });
  const del = useMutation({ mutationFn: () => endpoints.deletePersona(id!), onSuccess: () => { invalidate(keys.personas); nav("/personas"); } });
  const [previewText, setPreviewText] = useState("Okay, so, I got charged twice this month. Same invoice, FH-4419. I need that fixed today.");
  const previewRef = useRef<HTMLAudioElement | null>(null);
  const preview = useMutation({
    mutationFn: async () => { const b = await endpoints.ttsPreview({ text: previewText, voice: form!.voice, line: form!.line }); const url = URL.createObjectURL(b); previewRef.current?.pause(); const a = new Audio(url); previewRef.current = a; await a.play(); return new Promise<number>((res) => { a.onended = () => res(a.duration); }); },
  });
  useHotkeys({ "Ctrl+s": () => save.mutate(), "Ctrl+p": () => preview.mutate() });

  // The server renders it, so this is the caller's real prompt and not a retelling of it.
  const [promptForm, setPromptForm] = useState(form);
  useEffect(() => { const t = setTimeout(() => setPromptForm(form), 300); return () => clearTimeout(t); }, [form]);
  const prompt = useQuery({ queryKey: ["personaPrompt", promptForm], queryFn: () => endpoints.personaPromptPreview(promptForm!), enabled: !!promptForm, placeholderData: keepPreviousData });
  const title = isNew ? "New persona" : q.data?.name ?? "…";
  return (
    <Shell crumbs={[<Link key="p" to="/personas">Personas</Link>, title]} keys={[["Save", "Ctrl S"], ["Preview voice", "Ctrl P"]]}>
      <div className="page">
        {!isNew && q.isPending && <Skeleton rows={8} />}
        {!isNew && q.isError && <Note kind="err">Could not load this persona. {(q.error as Error).message}</Note>}
        {form && (
          <>
            <div className="page-head">
              <div><h1>{title}</h1><div className="page-sub">{!isNew && q.data ? (q.data.usage.calls ? <>Used in {q.data.usage.calls} calls. Pass rate with this persona <PassRate counts={q.data.usage.counts} bold={false} />.</> : "Not used in any calls yet.") : "Describe behaviour, not backstory. The caller model gets this verbatim."}</div></div>
              <div className="actions">
                {save.isError && <span style={{ color: "var(--fail)", fontSize: 12 }}>{(save.error as Error).message}</span>}
                <span className="faint">{dirty ? "unsaved changes" : savedAt ? "saved just now" : ""}</span>
                {!isNew && <Button variant="danger" onClick={() => { if (confirm("Delete this persona? Past calls keep their snapshot.")) del.mutate(); }}>Delete</Button>}
                <Button variant="primary" onClick={() => save.mutate()} disabled={save.isPending}>Save <Kbd>Ctrl</Kbd><Kbd>S</Kbd></Button>
              </div>
            </div>
            <div className={s.builder}>
              <div className="stack">
                <Panel title="Who they are">
                  <div className="stack">
                    <div className="form-row">
                      <Field label="Name" style={{ gridColumn: "span 2" }}><input className={`in ${!form.name.trim() && save.isError ? "err" : ""}`} value={form.name} onChange={(e) => up("name", e.target.value)} /></Field>
                      <Field label="Age"><input className="in" type="number" value={form.age ?? ""} onChange={(e) => up("age", e.target.value ? +e.target.value : null)} style={{ width: 80 }} /></Field>
                      <Field label="Temperament"><select className="in" value={form.temperament} onChange={(e) => up("temperament", e.target.value)}>{["calm", "neutral", "friendly", "anxious", "confused", "irritated", "impatient", "angry"].map((t) => <option key={t}>{t}</option>)}</select></Field>
                    </div>
                    <Field label="Summary" help="One line, shown in lists."><input className="in" value={form.summary} onChange={(e) => up("summary", e.target.value)} /></Field>
                    <Field label="Speaking style" help="Goes into the caller LLM's instructions verbatim."><textarea className="in" value={form.speakingStyle} onChange={(e) => up("speakingStyle", e.target.value)} /></Field>
                    <div className="form-row" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
                      <Scale label="Patience" value={form.patience} onChange={(v) => up("patience", v)} low="hangs up fast" high="waits it out" />
                      <Scale label="Verbosity" value={form.verbosity} onChange={(v) => up("verbosity", v)} low="one word" high="rambles" />
                      <Scale label="Interruption tendency" value={interruptionScale(form.interruptionRate)} onChange={(v) => up("interruptionRate", interruptionRate(v))} low="never" high="constantly" />
                      <Scale label="Tech savviness" value={form.techSavviness} onChange={(v) => up("techSavviness", v)} low="what is DNS" high="sysadmin" />
                    </div>
                    <Note kind="info">Interruption 5 means the caller bot starts talking as soon as it has a reply, even mid-sentence from the bot. In live mode this exercises barge-in. In turn-based mode it is ignored and the call is labelled so.</Note>
                  </div>
                </Panel>
                <Panel title="Voice">
                  <div className="stack">
                    <div className="form-row">
                      <Field label="TTS voice"><select className="in" value={`${form.voice.provider ?? ""}|${form.voice.voiceId}`} onChange={(e) => { const [provider, voiceId] = e.target.value.split("|"); up("voice", { ...form.voice, provider: provider || null, voiceId }); }}>
                        {voices.data?.items.map((v) => <option key={`${v.provider}|${v.voiceId}`} value={`${v.provider}|${v.voiceId}`}>{v.label}</option>)}
                        {!voices.data?.items.some((v) => v.voiceId === form.voice.voiceId) && <option value={`${form.voice.provider ?? ""}|${form.voice.voiceId}`}>{voiceLabel(form.voice.voiceId)}</option>}
                      </select></Field>
                      <Field label="Delivery hint" help="Free text for providers that take one."><input className="in" value={form.voice.style ?? ""} onChange={(e) => up("voice", { ...form.voice, style: e.target.value || null })} /></Field>
                    </div>
                    <Field label="Preview line" help={preview.isPending ? <HealthDot state="loading">rendering through the active TTS with the line condition applied</HealthDot> : preview.isSuccess ? <HealthDot state="ok">played {preview.data.toFixed(1)} s, {form.line.noise} noise at {form.line.snrDb} dB SNR, {form.line.codec}</HealthDot> : preview.isError ? <span style={{ color: "var(--fail)" }}>{(preview.error as Error).message}</span> : "Renders through the active TTS server with the line condition below applied. About 1 s."}>
                      <div style={{ display: "flex", gap: 8 }}><input className="in" value={previewText} onChange={(e) => setPreviewText(e.target.value)} /><Button onClick={() => preview.mutate()} disabled={preview.isPending}><Icon name="play" /> Preview <Kbd>Ctrl</Kbd><Kbd>P</Kbd></Button></div>
                    </Field>
                  </div>
                </Panel>
                <Panel title="Line conditions" meta="applied to this persona's audio before it reaches the bot">
                  <div className="form-row">
                    <Field label="Codec"><select className="in" value={form.line.codec} onChange={(e) => upLine("codec", e.target.value as LineConditions["codec"])}><option value="mulaw_8k">8 kHz mu-law (phone)</option><option value="none">16 kHz clean</option></select></Field>
                    <Field label="Background noise"><select className="in" value={form.line.noise} onChange={(e) => upLine("noise", e.target.value as LineConditions["noise"])}>{["none", "office", "street", "cafe", "car", "tv", "hum"].map((n) => <option key={n}>{n}</option>)}</select></Field>
                    <Field label="SNR (dB)" help="lower is noisier"><input className="in" type="number" value={form.line.snrDb} onChange={(e) => upLine("snrDb", +e.target.value)} style={{ width: 90 }} /></Field>
                    <Field label="Packet loss (%)"><input className="in" type="number" min={0} max={30} value={form.line.packetLossPct} onChange={(e) => upLine("packetLossPct", +e.target.value)} style={{ width: 80 }} /></Field>
                    <Field label="One-way latency (ms)"><input className="in" type="number" value={form.line.oneWayLatencyMs} onChange={(e) => upLine("oneWayLatencyMs", +e.target.value)} style={{ width: 90 }} /></Field>
                  </div>
                </Panel>
              </div>
              <div className="stack sticky-col">
                <Panel title="How this reads to the caller model"><div className={s.personaPrev}>{prompt.data?.prompt ?? (prompt.isError ? (prompt.error as Error).message : "…")}</div></Panel>
                {!isNew && q.data && (
                  <Panel flush title="Recent calls as this persona" actions={<Link className="faint" to={`/calls?persona=${id}`}>all</Link>}>
                    {q.data.recentCalls.length === 0 ? <Empty title="No calls yet">Calls made as this persona land here.</Empty> : <table className={`t ${s.sideCalls}`}><tbody>{q.data.recentCalls.map((c) => <tr key={c.id} tabIndex={0} onClick={() => nav(`/calls/${c.id}`)} onKeyDown={(e) => e.key === "Enter" && nav(`/calls/${c.id}`)}><td className="id">{c.id}</td><td>{c.verdict ? <VerdictBadge v={c.verdict} /> : <VerdictBadge v="evaluating" />}</td><td className={s.ell} title={c.scenarioTitle ?? undefined}>{c.scenarioTitle}</td><td><ModeBadge mode={c.mode} /></td></tr>)}</tbody></table>}
                  </Panel>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </Shell>
  );
}
