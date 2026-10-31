import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Shell } from "@/components/Shell";
import { Button, Empty, Field, Kbd, ModeBadge, Note, Panel, PassRate, QueryState, Skeleton, VerdictBadge } from "@/components/ui";
import { keys, useCustomers, useInvalidate, usePersonas, useScenario, useScenarios } from "@/api/queries";
import { endpoints } from "@/api/client";
import { useHotkeys } from "@/hooks/useHotkeys";
import { toast } from "@/store/toast";
import { ApiError } from "@/api/transport";
import type { Beat, Rule, Scenario, ScenarioDraft, ScenarioInput, StateExpectation } from "@/api/types";
import s from "./screens.module.css";

export function ScenariosScreen() {
  const q = useScenarios();
  return (
    <Shell crumbs={["Scenarios"]} actions={<Link to="/scenarios/new" className="btn sm primary">New scenario</Link>}>
      <div className="page">
        <div className="page-head"><div><h1>Scenarios</h1><div className="page-sub">What the caller is calling about, the facts it may use, the beats it must hit, and how the bot is judged.</div></div></div>
        <Panel flush>
          <QueryState q={q} what="scenarios" skeleton={<div style={{ padding: 12 }}><Skeleton rows={6} /></div>}>
            {(d) => d.items.length === 0 ? <Empty title="No scenarios yet"><Link to="/scenarios/new">Draft one with AI</Link> or seed from <span className="mono">seeds/scenarios</span>.</Empty> : d.items.map((sc) => (
              <Link key={sc.id} to={`/scenarios/${sc.id}`} className={s.listCard}>
                <span><b>{sc.title}</b><small>{sc.description || sc.goal}</small></span>
                <span className="faint" style={{ fontSize: 11 }}>{sc.tags.join(", ")}<small>{sc.must.length} must, {sc.mustNot.length} must not, {sc.beats.length} beats</small></span>
                <span style={{ fontSize: 12 }}>{sc.usage.calls ? <PassRate counts={sc.usage.counts} /> : <span className="faint">no calls</span>}<small>{sc.usage.calls} calls, v{sc.version}</small></span>
                <span className="faint mono" style={{ fontSize: 10 }}>{sc.id}</span>
              </Link>
            ))}
          </QueryState>
        </Panel>
      </div>
    </Shell>
  );
}

const blank = (): ScenarioInput => ({ title: "", description: "", goal: "", customerId: null, facts: {}, beats: [], must: [], mustNot: [], expectedState: [], maxTurns: 24, pronunciationWatch: [], tags: [] });
const nid = (p: string) => `${p}_${Math.random().toString(36).slice(2, 8)}`;

/* Editor state is keyed by resource id, so jumping from one scenario to another never carries the previous form over. */
export function ScenarioEditorScreen() {
  const { id } = useParams();
  return <ScenarioEditor key={id ?? "new"} id={id} />;
}

function ScenarioEditor({ id }: { id: string | undefined }) {
  const isNew = !id;
  const q = useScenario(id);
  const customers = useCustomers();
  const personas = usePersonas();
  const nav = useNavigate();
  const invalidate = useInvalidate();
  const [form, setForm] = useState<ScenarioInput | null>(isNew ? blank() : null);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (q.data && !form) setForm(q.data); }, [q.data, form]);
  const up = <K extends keyof ScenarioInput>(k: K, v: ScenarioInput[K]) => { setForm((f) => (f ? { ...f, [k]: v } : f)); setDirty(true); };
  const [facts, setFacts] = useState<[string, string][] | null>(null);
  useEffect(() => { if (form && !facts) setFacts(Object.entries(form.facts)); }, [form, facts]);
  const setFactsAnd = (fs: [string, string][]) => { setFacts(fs); up("facts", Object.fromEntries(fs.filter(([k]) => k.trim()))); };

  const validate = (f: ScenarioInput): string | null => {
    if (!f.title.trim()) return "Title is required.";
    if (!f.goal.trim()) return "The caller's goal is required.";
    if (!f.beats.some((b) => b.checkpoint)) return "At least one beat needs a checkpoint, or no run can be judged valid.";
    return null;
  };
  const save = useMutation({
    mutationFn: async () => { const err = validate(form!); if (err) throw new ApiError(400, { code: "invalid", message: err }, err); return isNew ? endpoints.createScenario(form!) : endpoints.updateScenario(id!, { ...(q.data as Scenario), ...form!, id: id! }); },
    onSuccess: (sc) => { invalidate(keys.scenarios, keys.scenario(sc.id)); setDirty(false); toast("Saved as a new version"); if (isNew) nav(`/scenarios/${sc.id}`, { replace: true }); },
  });
  const del = useMutation({ mutationFn: () => endpoints.deleteScenario(id!), onSuccess: () => { invalidate(keys.scenarios); nav("/scenarios"); } });
  const [dryPersona, setDryPersona] = useState<string>("");
  const dry = useMutation({ mutationFn: () => endpoints.dryRun(id!, dryPersona || null) });
  useHotkeys({ "Ctrl+s": () => save.mutate(), "Ctrl+Enter": () => addBeat(), "Ctrl+d": () => draft.mutate() });

  // AI draft
  const [desc, setDesc] = useState("");
  const [draftCustomer, setDraftCustomer] = useState<string>("");
  const [emptyOnly, setEmptyOnly] = useState(true);
  const draft = useMutation({ mutationFn: () => endpoints.draftScenario({ description: desc, customerId: draftCustomer || null, baseScenarioId: id ?? null }) });
  const acceptDraft = (d: ScenarioDraft, into: "this" | "new") => {
    const merge = (cur: ScenarioInput): ScenarioInput => {
      if (!emptyOnly || into === "new") return { ...d.scenario };
      const out = { ...cur };
      (Object.keys(d.scenario) as (keyof ScenarioInput)[]).forEach((k) => { const v = cur[k]; const empty = v == null || v === "" || (Array.isArray(v) && v.length === 0) || (typeof v === "object" && !Array.isArray(v) && Object.keys(v as object).length === 0); if (empty) (out as Record<string, unknown>)[k] = d.scenario[k]; });
      return out;
    };
    if (into === "new") { nav("/scenarios/new", { state: { draft: d.scenario } }); setForm({ ...d.scenario }); setFacts(Object.entries(d.scenario.facts)); setDirty(true); return; }
    setForm((f) => merge(f ?? blank())); setFacts(null); setDirty(true); draft.reset(); toast("Draft applied. Nothing is saved until you press Save.");
  };

  const addBeat = () => up("beats", [...(form?.beats ?? []), { id: nid("beat"), instruction: "", checkpoint: null, required: true }]);
  const upBeat = (i: number, b: Partial<Beat>) => up("beats", form!.beats.map((x, j) => (j === i ? { ...x, ...b } : x)));
  const moveBeat = (i: number, d: -1 | 1) => { const bs = [...form!.beats]; const j = i + d; if (j < 0 || j >= bs.length) return; [bs[i], bs[j]] = [bs[j], bs[i]]; up("beats", bs); };
  const upRule = (k: "must" | "mustNot", i: number, r: Partial<Rule>) => up(k, form![k].map((x, j) => (j === i ? { ...x, ...r } : x)));
  const upState = (i: number, e: Partial<StateExpectation>) => up("expectedState", form!.expectedState.map((x, j) => (j === i ? { ...x, ...e } : x)));

  const title = isNew ? "New scenario" : q.data?.title ?? "…";
  return (
    <Shell crumbs={[<Link key="s" to="/scenarios">Scenarios</Link>, title]} keys={[["Save", "Ctrl S"], ["Add beat", "Ctrl Enter"], ["Draft with AI", "Ctrl D"]]}>
      <div className="page">
        {!isNew && q.isPending && <Skeleton rows={8} />}
        {!isNew && q.isError && <Note kind="err">Could not load this scenario. {(q.error as Error).message}</Note>}
        {form && facts && (
          <>
            <div className="page-head">
              <div><h1>{title}</h1><div className="page-sub">{!isNew && q.data ? <>{q.data.usage.calls ? <>Used in {q.data.usage.calls} calls, <PassRate counts={q.data.usage.counts} bold={false} /> pass. </> : "Not used yet. "}Editing creates version {q.data.version + 1}; old calls keep version {q.data.version}.</> : "The caller bot sees the goal and the fact sheet. It never sees the rules."}</div></div>
              <div className="actions">
                {save.isError && <span style={{ color: "var(--fail)", fontSize: 12 }}>{(save.error as Error).message}</span>}
                <span className="faint">{dirty ? "unsaved changes" : ""}</span>
                {!isNew && <><select className="in" style={{ width: "auto", height: 28 }} value={dryPersona} onChange={(e) => setDryPersona(e.target.value)} aria-label="Dry run persona"><option value="">default caller</option>{personas.data?.items.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select><Button onClick={() => dry.mutate()} disabled={dry.isPending}>{dry.isPending ? "Running…" : "Dry run (text only)"}</Button></>}
                {!isNew && <Button variant="danger" onClick={() => { if (confirm("Delete this scenario? Past calls keep their snapshot.")) del.mutate(); }}>Delete</Button>}
                <Button variant="primary" onClick={() => save.mutate()} disabled={save.isPending}>Save <Kbd>Ctrl</Kbd><Kbd>S</Kbd></Button>
              </div>
            </div>
            {dry.isError && <Note kind="err" style={{ marginBottom: 12 }}>Dry run failed. {(dry.error as Error).message}</Note>}
            {dry.data && <Panel title="Dry run" meta={`${dry.data.model}, text only, reached ${dry.data.reachedCheckpoints.length} of ${form.beats.filter((b) => b.checkpoint).length} checkpoints`} actions={<Button size="sm" variant="ghost" onClick={() => dry.reset()}>Close</Button>} style={{ marginBottom: 12 }}>
              <div className={s.liveTurns} style={{ padding: 0 }}>{dry.data.turns.map((t, i) => <div key={i} className={`${s.lt} ${s[t.speaker]}`}><b>{t.speaker}</b><span>{t.text}</span></div>)}</div>
            </Panel>}
            <div className={s.builder}>
              <div className="stack">
                <Panel title="The issue">
                  <div className="stack">
                    <div className="form-row">
                      <Field label="Title" style={{ gridColumn: "span 2" }}><input className="in" value={form.title} onChange={(e) => up("title", e.target.value)} /></Field>
                      <Field label="Tags"><input className="in" value={form.tags.join(", ")} onChange={(e) => up("tags", e.target.value.split(",").map((x) => x.trim()).filter(Boolean))} /></Field>
                    </div>
                    <Field label="Description" help="For people. What the test is about and what correct behaviour looks like."><input className="in" value={form.description} onChange={(e) => up("description", e.target.value)} /></Field>
                    <Field label="What the caller wants, in their words" help="The caller bot sees this and the fact sheet. It does not see the rules below."><textarea className="in" value={form.goal} onChange={(e) => up("goal", e.target.value)} /></Field>
                    <div className="form-row">
                      <Field label="Mock customer" help={<>Each call gets a fresh copy of this customer's seeded state.{form.customerId && <> <a href="#" onClick={(e) => { e.preventDefault(); toast(customers.data?.items.find((c) => c.id === form.customerId)?.note ?? ""); }}>Seed note</a></>}</>}>
                        <select className={`in ${form.customerId && customers.data && !customers.data.items.some((c) => c.id === form.customerId) ? "err" : ""}`} value={form.customerId ?? ""} onChange={(e) => up("customerId", e.target.value || null)}>
                          <option value="">none</option>{customers.data?.items.map((c) => <option key={c.id} value={c.id}>{c.id} {c.name} ({c.plans.join(", ")})</option>)}
                          {form.customerId && customers.data && !customers.data.items.some((c) => c.id === form.customerId) && <option value={form.customerId}>{form.customerId} (not in seed)</option>}
                        </select></Field>
                      <Field label="Turn limit" help="Call ends as invalid if the caller has not reached its checkpoints by then."><input className="in" type="number" min={2} max={120} value={form.maxTurns} onChange={(e) => up("maxTurns", +e.target.value)} style={{ width: 80 }} /></Field>
                    </div>
                  </div>
                </Panel>
                <Panel title="Caller's fact sheet" meta="the caller may only use these facts" actions={<Button size="sm" variant="ghost" onClick={() => setFactsAnd([...facts, ["", ""]])}>Add fact</Button>}>
                  <div className="stack" style={{ gap: 4 }}>
                    {facts.length === 0 && <span className="faint" style={{ fontSize: 12 }}>No facts yet. Without them the caller invents details.</span>}
                    {facts.map(([k, v], i) => <div key={i} className={s.fact}><input className="in mono" value={k} aria-label="fact name" onChange={(e) => setFactsAnd(facts.map((f, j) => (j === i ? [e.target.value, f[1]] : f)))} /><input className="in" value={v} aria-label="fact value" onChange={(e) => setFactsAnd(facts.map((f, j) => (j === i ? [f[0], e.target.value] : f)))} /><button type="button" className={s.x} aria-label="remove" onClick={() => setFactsAnd(facts.filter((_, j) => j !== i))}>×</button></div>)}
                  </div>
                </Panel>
                <Panel title="Beats, in order" meta="checkpoints decide whether a run is valid" actions={<Button size="sm" variant="ghost" onClick={addBeat}>Add beat <Kbd>Ctrl</Kbd><Kbd>↵</Kbd></Button>}>
                  <div className={s.beats}>
                    {form.beats.length === 0 && <span className="faint" style={{ fontSize: 12 }}>No beats. Add at least one with a checkpoint.</span>}
                    {form.beats.map((b, i) => (
                      <div key={b.id} className={s.beat}>
                        <span className={s.n}>{i + 1}</span>
                        <div>
                          <input className="in" value={b.instruction} placeholder="What the caller should do or say" onChange={(e) => upBeat(i, { instruction: e.target.value })} />
                          <div className={s.meta}>
                            <label className="check" style={{ fontSize: 11 }}><input type="checkbox" checked={b.checkpoint != null} onChange={(e) => upBeat(i, { checkpoint: e.target.checked ? (b.checkpoint ?? "") : null })} /> checkpoint</label>
                            {b.checkpoint != null && <input className="in" type="text" style={{ flex: 1, minWidth: 200 }} placeholder="What proves this beat was reached" value={b.checkpoint} onChange={(e) => upBeat(i, { checkpoint: e.target.value })} />}
                            <label className="check" style={{ fontSize: 11 }}><input type="checkbox" checked={b.required} onChange={(e) => upBeat(i, { required: e.target.checked })} /> required</label>
                          </div>
                        </div>
                        <div className={s.ctl}><button type="button" aria-label="move up" onClick={() => moveBeat(i, -1)}>▲</button><button type="button" aria-label="move down" onClick={() => moveBeat(i, 1)}>▼</button><button type="button" aria-label="remove" onClick={() => up("beats", form.beats.filter((_, j) => j !== i))}>×</button></div>
                      </div>
                    ))}
                  </div>
                </Panel>
                <Panel title="Rules the bot is judged on" meta='the transcript judge quotes evidence for each. A failed critical "must not" always fails the call.' actions={<><Button size="sm" variant="ghost" onClick={() => up("must", [...form.must, { id: nid("rule"), text: "", severity: "major" }])}>Add must</Button><Button size="sm" variant="ghost" onClick={() => up("mustNot", [...form.mustNot, { id: nid("rule"), text: "", severity: "critical" }])}>Add must not</Button></>}>
                  <div className={s.rules}>
                    {(["must", "mustNot"] as const).flatMap((k) => form[k].map((r, i) => (
                      <div key={r.id} className={s.rule}>
                        <span className={`${s.k} ${k === "must" ? s.must : s.mustnot}`} title={r.id}>{k === "must" ? "must" : "not"} {r.id.replace(/^rule_/, "").slice(0, 10)}</span>
                        <input className="in" value={r.text} placeholder={k === "must" ? "The bot must…" : "The bot must not…"} onChange={(e) => upRule(k, i, { text: e.target.value })} />
                        <select className="in" value={r.severity} onChange={(e) => upRule(k, i, { severity: e.target.value as Rule["severity"] })}><option value="critical">critical</option><option value="major">major</option><option value="minor">minor</option></select>
                        <button type="button" className={s.x} aria-label="remove" onClick={() => up(k, form[k].filter((_, j) => j !== i))}>×</button>
                      </div>
                    )))}
                    {form.must.length + form.mustNot.length === 0 && <span className="faint" style={{ fontSize: 12 }}>No rules. The judge will have nothing to score.</span>}
                  </div>
                </Panel>
                <div className="grid cols-2">
                  <Panel title="Expected database changes" actions={<Button size="sm" variant="ghost" onClick={() => up("expectedState", [...form.expectedState, { id: nid("state"), table: "refunds", where: {}, op: "row_added", changes: {}, description: "" }])}>Add</Button>}>
                    <div className="stack" style={{ gap: 4 }}>
                      {form.expectedState.map((e, i) => (
                        <div key={e.id} className={s.state}>
                          <input className="in mono" value={e.table} aria-label="table" onChange={(ev) => upState(i, { table: ev.target.value })} />
                          <input className="in mono" value={kv(e.where)} aria-label="where" placeholder="where: invoice_id=FH-4419" onChange={(ev) => upState(i, { where: parseKv(ev.target.value) })} />
                          <select className="in" value={e.op} aria-label="op" onChange={(ev) => upState(i, { op: ev.target.value as StateExpectation["op"] })}><option value="row_added">row added</option><option value="changed">changed</option><option value="no_change">no change</option></select>
                          <input className="in mono" value={kv(e.changes)} aria-label="changes" placeholder={e.op === "changed" ? "status=paid" : "description"} onChange={(ev) => e.op === "changed" ? upState(i, { changes: parseKv(ev.target.value) }) : upState(i, { description: ev.target.value })} disabled={false} />
                          <button type="button" className={s.x} aria-label="remove" onClick={() => up("expectedState", form.expectedState.filter((_, j) => j !== i))}>×</button>
                        </div>
                      ))}
                      <div className="help">Checked by diffing the call's database copy against the seed. Unlisted changes are flagged.</div>
                    </div>
                  </Panel>
                  <Panel title="Pronunciation watch list">
                    <input className="in mono" value={form.pronunciationWatch.join(", ")} onChange={(e) => up("pronunciationWatch", e.target.value.split(",").map((x) => x.trim()).filter(Boolean))} />
                    <div className="help" style={{ marginTop: 4 }}>Tokens weighted heavily by the pronunciation evaluator. Numbers and names in the fact sheet are added automatically.</div>
                  </Panel>
                </div>
              </div>
              <div className="stack sticky-col">
                <Panel title="Draft with AI" meta={draft.data?.model ?? ""}>
                  <div className="stack">
                    <Field label="Describe the situation in plain words"><textarea className="in" style={{ minHeight: 90 }} value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Customer whose website got suspended for non-payment but they paid two days ago. Angry, wants it back up now." /></Field>
                    <Field label="Customer (optional)"><select className="in" value={draftCustomer} onChange={(e) => setDraftCustomer(e.target.value)}><option value="">let the model pick from the seed</option>{customers.data?.items.map((c) => <option key={c.id} value={c.id}>{c.name}: {c.note}</option>)}</select></Field>
                    <div className="btn-row"><Button variant="primary" onClick={() => draft.mutate()} disabled={draft.isPending || !desc.trim()}>Draft scenario <Kbd>Ctrl</Kbd><Kbd>D</Kbd></Button><label className="check"><input type="checkbox" checked={emptyOnly} onChange={(e) => setEmptyOnly(e.target.checked)} /> replace empty fields only</label></div>
                    {draft.isPending && <div><Skeleton rows={3} /><div className="faint" style={{ marginTop: 6 }}>Drafting, usually 5 to 10 s.</div></div>}
                    {draft.isError && <Note kind={(draft.error as ApiError).code === "model_not_ready" ? "warn" : "err"}>{(draft.error as Error).message} Your description is kept.</Note>}
                    {draft.data && (
                      <div>
                        <Note kind="info">Draft ready. Nothing is saved until you accept. Fields marked <b>guessed</b> need checking against the seed: {draft.data.guessedFields.join(", ")}.</Note>
                        <div style={{ marginTop: 8 }}>
                          <div className="label">Proposed</div>
                          <div className={s.draftIns}><b>Title</b> {draft.data.scenario.title}</div>
                          <div className={s.draftIns}><b>Goal</b> {draft.data.scenario.goal}</div>
                          <div className={s.draftIns}><b>Customer</b> {draft.data.scenario.customerId ?? "none"} {draft.data.guessedFields.includes("customerId") && <span className="tag">guessed</span>}</div>
                          <div className={s.draftIns}><b>Facts</b> {Object.entries(draft.data.scenario.facts).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
                          <div className={s.draftIns}><b>Beats</b> {draft.data.scenario.beats.map((b, i) => `${i + 1} ${b.instruction}${b.checkpoint ? " (checkpoint)" : ""}`).join(" · ")}</div>
                          <div className={s.draftIns}><b>Must</b> {draft.data.scenario.must.map((r) => r.text).join(" · ")}</div>
                          <div className={s.draftIns}><b>Must not</b> {draft.data.scenario.mustNot.map((r) => r.text).join(" · ")}</div>
                          <div className={s.draftIns}><b>Expected db</b> {draft.data.scenario.expectedState.map((e) => `${e.table} ${e.op} ${kv(e.where)}`).join(" · ")}</div>
                          <div className={s.draftIns}><b>Watch list</b> {draft.data.scenario.pronunciationWatch.join(", ")}</div>
                          {draft.data.notes && <div className="faint" style={{ marginTop: 4, fontSize: 12 }}>{draft.data.notes}</div>}
                        </div>
                        <div className="btn-row" style={{ marginTop: 8 }}>{!isNew && <Button variant="primary" onClick={() => acceptDraft(draft.data!, "new")}>Accept into a new scenario</Button>}<Button variant={isNew ? "primary" : "default"} onClick={() => acceptDraft(draft.data!, "this")}>Accept into this one</Button><Button variant="ghost" onClick={() => draft.reset()}>Discard</Button></div>
                      </div>
                    )}
                  </div>
                </Panel>
                {!isNew && q.data && (
                  <Panel flush title="Recent calls" actions={<Link className="faint" to={`/calls?scenario=${id}`}>all</Link>}>
                    {q.data.recentCalls.length === 0 ? <Empty title="No calls yet" /> : <table className="t"><tbody>{q.data.recentCalls.map((c) => <tr key={c.id} tabIndex={0} onClick={() => nav(`/calls/${c.id}`)}><td className="id">{c.id}</td><td>{c.verdict ? <VerdictBadge v={c.verdict} /> : <VerdictBadge v="evaluating" />}</td><td>{c.personaName}</td><td><ModeBadge mode={c.mode} /></td></tr>)}</tbody></table>}
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

const kv = (o: Record<string, unknown>) => Object.entries(o).map(([k, v]) => `${k}=${v}`).join(", ");
const parseKv = (sv: string): Record<string, unknown> => Object.fromEntries(sv.split(",").map((p) => p.trim()).filter(Boolean).map((p) => { const [k, ...r] = p.split("="); const v = r.join("=").trim(); return [k.trim(), /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v]; }));
