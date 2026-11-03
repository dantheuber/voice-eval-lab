import { Link, useNavigate } from "react-router-dom";
import { Shell } from "@/components/Shell";
import { Button, Empty, HealthDot, Kbd, ModeBadge, Note, Panel, PassRate, QueryState, RateBar, RateLegend, Skeleton, Untested } from "@/components/ui";
import { useOverview, useProfiles } from "@/api/queries";
import { isUntested } from "@/lib/untested";
import { isAdvisoryEvaluator } from "@/lib/evaluators";
import { endpoints } from "@/api/client";
import { ciText, valid } from "@/lib/stats";
import { evalName, fmtAgo, fmtWhen } from "@/lib/format";
import s from "./screens.module.css";
import type { Overview } from "@/api/types";

export function OverviewScreen() {
  const q = useOverview();
  const nav = useNavigate();
  const goNext = async () => { const r = await endpoints.nextReview(); if (r.callId) nav(`/calls/${r.callId}`); };
  return (
    <Shell crumbs={["Overview"]} actions={<Button variant="primary" size="sm" onClick={goNext}>Start reviewing <Kbd>g</Kbd><Kbd>r</Kbd></Button>}>
      <div className="page">
        <QueryState q={q} what="the overview" skeleton={<div className={s.ovGrid}>{[0, 1, 2, 3].map((i) => <div key={i} className="panel stat"><Skeleton rows={3} /></div>)}</div>}>
          {(d) => <OverviewBody d={d} />}
        </QueryState>
      </div>
    </Shell>
  );
}

function OverviewBody({ d }: { d: Overview }) {
  const nav = useNavigate();
  const profiles = useProfiles();
  const loading = d.health.find((h) => h.state === "loading");
  const down = d.health.filter((h) => h.state === "down");
  const v = valid(d.counts);
  const total = v + d.counts.invalid;
  const paceMin = d.reviewPaceSPerFinding ? Math.round((d.reviewPaceSPerFinding * d.openFindingsTotal) / 60) : null;
  const noCalls = total === 0 && d.recentRuns.length === 0;
  const maxH = Math.max(1, ...d.callsPerHour);
  return (
    <>
      {loading && <Note kind="warn" style={{ marginBottom: 12 }}><b>{loading.name} is loading</b> ({loading.model}{loading.elapsedS != null ? `, ${loading.elapsedS} s so far` : ""}{loading.usualS != null ? `, usually under ${loading.usualS} s` : ""}). Calls can run, but evals queue until it is ready. <Link to="/settings">Model status</Link></Note>}
      {down.length > 0 && <Note kind="err" style={{ marginBottom: 12 }}><b>{down.map((h) => h.name).join(", ")} {down.length > 1 ? "are" : "is"} down.</b> {down[0].detail} <Link to="/settings">Restart from settings</Link></Note>}
      {noCalls ? (
        <Panel>
          <Empty title="No calls yet">Launch a run against the seeded scenarios, or talk to the bot yourself.
            <div className="btn-row" style={{ justifyContent: "center", marginTop: 12 }}><Button variant="primary" onClick={() => nav("/runs/new")}>Launch a run</Button><Button onClick={() => nav("/talk")}>Talk to the bot</Button></div></Empty>
        </Panel>
      ) : (
        <div className={s.ovGrid}>
          <div className="panel stat">
            <span className="k">Pass rate, {d.window}, {d.profile}</span>
            {v ? <span className="n">{Math.round((100 * d.counts.pass) / v)}%<small>{d.counts.pass} of {v} valid calls</small></span> : <span className="n">--<small>no valid calls</small></span>}
            <span className="d">95% interval {ciText(d.counts.pass, v) || "n/a"}. {d.counts.invalid} invalid runs excluded.</span>
            <div style={{ marginTop: 8 }}><RateBar counts={d.counts} /></div>
            <div style={{ marginTop: 6 }}><RateLegend counts={d.counts} /></div>
          </div>
          <div className="panel stat">
            <span className="k">Needs a human</span>
            <span className="n">{d.queueTotal}<small>calls, {d.openFindingsTotal} open findings</small></span>
            <span className="d">{d.queue[0] ? `Oldest is from ${fmtAgo(d.queue[0].createdAt)}.` : "Nothing waiting."}{paceMin != null && d.queueTotal ? ` At your usual pace, about ${Math.max(1, paceMin)} minutes.` : ""}</span>
            <div style={{ marginTop: "auto", paddingTop: 8 }}>{d.queueTotal ? <Button size="sm" onClick={async () => { const r = await endpoints.nextReview(); if (r.callId) nav(`/calls/${r.callId}`); }}>Review next <Kbd>]</Kbd></Button> : null}</div>
          </div>
          <div className="panel stat">
            <span className="k">Ran recently</span>
            <span className="n">{d.recentRuns.length}<small>runs, {d.recentRuns.reduce((a, r) => a + r.calls, 0)} calls</small></span>
            <span className="d">{d.runningRuns ? `${d.runningRuns} run in progress.` : "No run in progress."}{d.recentRuns[0]?.endedAt ? ` Last finished ${fmtAgo(d.recentRuns[0].endedAt)}.` : ""}</span>
            <div className={s.sparkbar} style={{ marginTop: "auto" }} role="img" aria-label="Calls per hour over the last 24 hours">{d.callsPerHour.map((n, i) => <i key={i} style={{ height: `${Math.max(6, (n / maxH) * 100)}%` }} title={`${n} calls`} />)}</div>
            <div className="faint" style={{ fontSize: 10, display: "flex", justifyContent: "space-between" }}><span>24 h ago</span><span>now</span></div>
          </div>
          <Panel flush className={s.queue} title="Needs a human" meta={<span className="tag mono">{d.queueTotal}</span>} actions={<Link to="/calls?verdict=needs_review" className="faint">all</Link>}>
            {d.queue.length === 0 && <Empty title="Nothing to review">New findings land here as evals finish.</Empty>}
            {d.queue.map((qi) => (
              <Link key={qi.callId} className={s.queueItem} to={`/calls/${qi.callId}`}>
                <span className="a">{qi.scenarioTitle ?? "unknown scenario"} <span className="muted">{qi.personaName ?? ""}</span></span>
                <span className="c"><span className="tag mono">{qi.openFindings} open</span></span>
                <span className="b">{qi.callId} · {qi.why}</span>
              </Link>
            ))}
          </Panel>
          <Panel flush className={s.runs} title="Recent runs" actions={<Link to="/runs/new" className="btn sm">New run</Link>}>
            {d.recentRuns.length === 0 && <Empty title="No runs yet"><Link to="/runs/new">Launch a run</Link> to fill this in.</Empty>}
            {d.recentRuns.length > 0 && <div style={{ overflow: "auto" }}>
              <table className="t">
                <thead><tr><th>Run</th><th>When</th><th>Profile</th><th>Mode</th><th>What</th><th className="num">Calls</th><th style={{ width: 170 }}>Outcome</th><th className="num">Pass</th></tr></thead>
                <tbody>{d.recentRuns.map((r) => (
                  <tr key={r.id} tabIndex={0} onClick={() => nav(r.status === "running" ? `/runs/${r.id}` : `/calls?run=${r.id}`)} onKeyDown={(e) => e.key === "Enter" && nav(`/calls?run=${r.id}`)}>
                    <td className="id">{r.id}</td><td className="ts">{fmtWhen(r.startedAt)}</td><td className="mono">{r.profile} {isUntested(r, profiles.data?.items) && <Untested />}</td><td><ModeBadge mode={r.mode} /></td><td>{r.label}{r.status === "running" ? <span className="v running" style={{ marginLeft: 6 }}>running</span> : null}</td>
                    <td className="num">{r.calls}</td><td><RateBar counts={r.counts} /></td><td className="num"><PassRate counts={r.counts} /></td>
                  </tr>))}</tbody>
              </table>
            </div>}
          </Panel>
        </div>
      )}
      <div className={s.ovBottom}>
        <Panel flush title="Pass rate by scenario" meta={`7 days, ${d.profile}`}>
          {d.scenarioRates.length === 0 && <Empty title="No calls in this window">Pass rates by scenario appear once calls finish on this profile.</Empty>}
          {d.scenarioRates.map((r) => { const vv = valid(r.counts); return (
            <Link key={r.scenarioId} className={s.rateRow} to={`/calls?scenario=${r.scenarioId}`} title={r.scenarioTitle}>
              <span><span className="muted">{r.scenarioTitle}</span></span><RateBar counts={r.counts} />
              <span className={s.pct}>{vv ? `${Math.round((100 * r.counts.pass) / vv)}%` : "--"} <span className="ci" style={{ fontWeight: 400 }}>{ciText(r.counts.pass, vv)}</span></span><span className={s.n}>n={vv}</span>
            </Link>); })}
          <div style={{ padding: "8px 12px 2px" }}><RateLegend /></div>
        </Panel>
        <Panel flush title="Models" actions={<Link to="/settings" className="faint">settings</Link>}>
          <table className="t"><tbody>{d.health.map((h) => <tr key={h.role}><td style={{ width: 130 }}>{h.name}</td><td className="mono faint">{h.model}</td><td className="right"><HealthDot state={h.state}>{h.detail}{h.state === "loading" && h.elapsedS != null ? `, ${h.elapsedS} s` : ""}</HealthDot></td></tr>)}</tbody></table>
        </Panel>
        <Panel flush title="Most common open findings" meta="last 7 days">
          {d.topOpenFindings.length === 0 ? <Empty title="No open findings">Findings waiting on a decision are counted here by title.</Empty> : <table className="t"><tbody>{d.topOpenFindings.map((f) => <tr key={f.title}><td>{f.title}</td><td className="faint">{evalName(f.evaluator)}{isAdvisoryEvaluator(f.evaluator) ? ", advisory" : ""}</td><td className="num">{f.count}</td></tr>)}</tbody></table>}
        </Panel>
      </div>
    </>
  );
}
