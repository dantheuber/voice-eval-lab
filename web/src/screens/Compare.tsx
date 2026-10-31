import { Link, useSearchParams } from "react-router-dom";
import { Shell } from "@/components/Shell";
import { Empty, ModeBadge, Note, Panel, PassFrac, PassRate, QueryState, RateBar, Skeleton, Untested } from "@/components/ui";
import { useCompare, usePersonas, useProfiles, useRuns, useScenarios } from "@/api/queries";
import { ciText, intervalsOverlap, rateDirection, valid } from "@/lib/stats";
import { isUntested } from "@/lib/untested";
import type { Compare, Frac, RunSummary } from "@/api/types";
import s from "./screens.module.css";

export function CompareScreen() {
  const [sp, setSp] = useSearchParams();
  const runs = useRuns();
  const a = sp.get("a") ?? "", b = sp.get("b") ?? "";
  const cmp = useCompare(a || undefined, b || undefined);
  const set = (k: "a" | "b", v: string) => { const n = new URLSearchParams(sp); if (v) n.set(k, v); else n.delete(k); setSp(n, { replace: true }); };
  const options = (runs.data?.items ?? []).filter((r) => r.status !== "queued");
  return (
    <Shell crumbs={["Compare", a && b ? `${a} vs ${b}` : "pick two runs"]}>
      <div className="page">
        <div className="page-head"><div><h1>Compare runs</h1><div className="page-sub">Two batches side by side. Per-criterion rows are where real regressions show; headline pass rates from small runs mostly overlap.</div></div>
          <div className="actions">
            <select className="in" style={{ width: "auto" }} value={a} onChange={(e) => set("a", e.target.value)} aria-label="Run A"><option value="">Run A</option>{options.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.profile} · {r.label}</option>)}</select>
            <span className="faint">vs</span>
            <select className="in" style={{ width: "auto" }} value={b} onChange={(e) => set("b", e.target.value)} aria-label="Run B"><option value="">Run B</option>{options.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.profile} · {r.label}</option>)}</select>
          </div></div>
        {!a || !b ? <Panel><Empty title="Pick two runs">Choose runs with the same scenarios, for example the same batch on two model profiles or two bot versions.</Empty></Panel>
          : <QueryState q={cmp} what="the comparison" skeleton={<Skeleton rows={8} />}>{(d) => <CompareBody d={d} />}</QueryState>}
      </div>
    </Shell>
  );
}

/* Headline note computed from the two intervals, not a fixed threshold. */
export function headlineNote(a: Frac, b: Frac): { kind: "warn" | "info" | "plain"; text: string } {
  const pa = a.n ? Math.round((100 * a.pass) / a.n) : null, pb = b.n ? Math.round((100 * b.pass) / b.n) : null;
  if (pa == null || pb == null) return { kind: "plain", text: "One run has no valid calls yet, so the headline pass rates cannot be compared." };
  const diff = pb - pa;
  const overlap = intervalsOverlap(a, b);
  const ivs = `${ciText(a.pass, a.n)} (n=${a.n}) vs ${ciText(b.pass, b.n)} (n=${b.n})`;
  if (overlap) return { kind: "warn", text: `The 95% intervals overlap, ${ivs}, so the ${Math.abs(diff)}-point headline difference is within noise at these sample sizes. Per-criterion rows below are where real regressions show up.` };
  return { kind: "info", text: `The 95% intervals do not overlap, ${ivs}. The ${Math.abs(diff)}-point ${diff > 0 ? "improvement" : "regression"} is unlikely to be noise.` };
}

function CompareBody({ d }: { d: Compare }) {
  const personas = usePersonas();
  const scenarios = useScenarios();
  const profiles = useProfiles();
  const pct = (k: number, n: number) => (n ? Math.round((100 * k) / n) : null);
  const delta = (x: number | null, y: number | null, lowerIsBetter = false, suffix = "") => {
    if (x == null || y == null) return <span className="delta same">not comparable</span>;
    const dd = +(y - x).toFixed(1);
    const cls = dd === 0 ? "same" : (dd > 0) !== lowerIsBetter ? "better" : "worse";
    return <span className={`delta ${cls}`}>{dd > 0 ? "+" : ""}{dd}{suffix}{cls === "worse" ? " ▼" : cls === "better" ? " ▲" : ""}</span>;
  };
  const col = (r: RunSummary) => { const v = valid(r.counts); return (
    <div className="panel stat"><span className="k" style={{ textTransform: "none" }}>{r.id} · <span className="mono">{r.profile}</span> <ModeBadge mode={r.mode} /> {isUntested(r, profiles.data?.items) && <Untested />}</span>
      <span className="n">{v ? `${Math.round((100 * r.counts.pass) / v)}%` : "--"}<small>{r.counts.pass} of {v} valid, {ciText(r.counts.pass, v) || "no interval"}</small></span><RateBar counts={r.counts} />
      <span className="d">{r.label}{r.counts.invalid ? `, ${r.counts.invalid} invalid excluded` : ""}{r.status === "running" ? ", still running (partial)" : ""}</span></div>); };
  const fa: Frac = { pass: d.a.counts.pass, n: valid(d.a.counts) }, fb: Frac = { pass: d.b.counts.pass, n: valid(d.b.counts) };
  const note = headlineNote(fa, fb);
  const anyTurn = d.a.mode === "turn_based" || d.b.mode === "turn_based";
  const pName = (id: string) => personas.data?.items.find((p) => p.id === id)?.name.split(" ").slice(0, 2).join(" ") ?? id;
  const sTitle = (id: string) => scenarios.data?.items.find((x) => x.id === id)?.title ?? id;
  const cellPersonas = [...new Set(d.cells.map((c) => c.personaId))];
  const cellScenarios = [...new Set(d.cells.map((c) => c.scenarioId))];
  const worst = d.worst ? d.criteria.find((c) => c.criterionId === d.worst!.criterionId) : null;
  return (
    <>
      <Note kind={note.kind} style={{ marginBottom: 12 }}>{note.text}{anyTurn ? " One run is turn-based, so timing rows are not comparable." : ""}</Note>
      {d.droppedScenarioIds.length > 0 && <Note kind="plain" style={{ marginBottom: 12 }}>Only scenarios in both runs are compared. Dropped: {d.droppedScenarioIds.map(sTitle).join(", ")}.</Note>}
      <div className={s.cmp} style={{ marginBottom: 12 }}>{col(d.a)}{col(d.b)}</div>
      <div className={s.cmpBody}>
        <Panel flush title="By rubric criterion" meta="calls passing each rule, of valid calls">
          <table className="t"><thead><tr><th>Rule</th><th className="num">{d.a.id}</th><th className="num">{d.b.id}</th><th className="num">Δ pts</th></tr></thead>
            <tbody>{d.criteria.map((c) => { const pa = pct(c.a.pass, c.a.n), pb = pct(c.b.pass, c.b.n); return <tr key={c.criterionId}><td><span className="mono faint">{c.kind === "must_not" ? "not" : c.kind}</span> {c.text}</td><td className="num"><PassFrac pass={c.a.pass} n={c.a.n} bold={false} compact /></td><td className="num"><PassFrac pass={c.b.pass} n={c.b.n} bold={false} compact /></td><td className="num">{delta(pa, pb)}</td></tr>; })}</tbody></table>
          {worst && d.worst && <div style={{ margin: 12 }}><Note kind="info"><b>Regressed:</b> "{worst.text}" dropped {Math.abs(d.worst.deltaPts)} points on {d.b.profile}. {d.worst.callIds.length} call{d.worst.callIds.length === 1 ? "" : "s"} affected. <Link to={`/calls?run=${d.b.id}&verdict=fail`}>Open those calls</Link>.</Note></div>}
        </Panel>
        <div className="stack">
          <Panel flush title="Other measures">
            <table className="t"><thead><tr><th>Measure</th><th className="num">{d.a.id}</th><th className="num">{d.b.id}</th><th className="num">Δ</th></tr></thead>
              <tbody>{d.measures.map((m) => <tr key={m.key}><td>{m.label}</td><td className="num mono">{m.a == null ? <span className="faint">n/a{m.reason ? `, ${m.reason}` : ""}</span> : m.a}</td><td className="num mono">{m.b == null ? <span className="faint">n/a{m.reason ? `, ${m.reason}` : ""}</span> : m.b}</td><td className="num">{m.comparable ? delta(m.a, m.b, !!m.lowerIsBetter) : <span className="delta same">{m.reason ?? "not comparable"}</span>}</td></tr>)}</tbody></table>
          </Panel>
          <Panel flush title="By scenario × persona" meta="pass of valid reps, A → B, direction by rate">
            <div style={{ overflow: "auto" }}><table className={s.matrix}><thead><tr><th></th>{cellPersonas.map((p) => <th key={p} className={s.p}>{pName(p)}</th>)}</tr></thead>
              <tbody>{cellScenarios.map((sc) => <tr key={sc}><td className={s.s}>{sTitle(sc)}</td>{cellPersonas.map((p) => { const c = d.cells.find((x) => x.scenarioId === sc && x.personaId === p); if (!c) return <td key={p} className="faint">–</td>; const dir = rateDirection(c.a, c.b); return <td key={p} title={`${pct(c.a.pass, c.a.n) ?? "n/a"}% → ${pct(c.b.pass, c.b.n) ?? "n/a"}%`}><span className="mono">{c.a.pass}/{c.a.n}→{c.b.pass}/{c.b.n}</span> {dir === "better" ? <span className="delta better">▲</span> : dir === "worse" ? <span className="delta worse">▼</span> : dir === null ? <span className="faint">n/a</span> : ""}</td>; })}</tr>)}</tbody></table></div>
          </Panel>
          <Panel flush title="Headline">
            <table className="t"><tbody><tr><td>{d.a.id}</td><td className="num"><PassRate counts={d.a.counts} /></td></tr><tr><td>{d.b.id}</td><td className="num"><PassRate counts={d.b.counts} /></td></tr></tbody></table>
          </Panel>
        </div>
      </div>
    </>
  );
}
