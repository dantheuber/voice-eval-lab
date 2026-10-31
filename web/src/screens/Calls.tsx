import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { createColumnHelper, flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type SortingState } from "@tanstack/react-table";
import { Shell } from "@/components/Shell";
import { Button, Empty, ModeBadge, Panel, QueryState, Skeleton, Untested, VerdictBadge } from "@/components/ui";
import { useCalls, usePersonas, useProfiles, useScenarios } from "@/api/queries";
import { useHotkeys } from "@/hooks/useHotkeys";
import { isUntested } from "@/lib/untested";
import { fmtClock, verdictLabel } from "@/lib/format";
import type { CallMode, CallSummary, CallsQuery, Verdict } from "@/api/types";
import s from "./screens.module.css";

const col = createColumnHelper<CallSummary>();
const VERDICTS: Verdict[] = ["pass", "fail", "needs_review", "invalid"];
const KEYS: [string, string][] = [["Move selection", "j k"], ["Open selected call", "Enter"], ["Open in review", "o"], ["Clear filters", "Esc"]];

export function CallsScreen() {
  const [sp, setSp] = useSearchParams();
  const nav = useNavigate();
  const q: CallsQuery = useMemo(() => ({
    verdict: sp.getAll("verdict") as Verdict[], scenario: sp.get("scenario") ?? undefined, persona: sp.get("persona") ?? undefined,
    mode: (sp.get("mode") as CallMode) ?? undefined, profile: sp.get("profile") ?? undefined, run: sp.get("run") ?? undefined, openOnly: sp.get("open") === "1",
    cursor: sp.get("cursor") ?? undefined, limit: sp.get("limit") ? Number(sp.get("limit")) : undefined,
  }), [sp]);
  // Cursors are opaque, so previous pages are a stack kept while the filters stay the same.
  const [cursorStack, setCursorStack] = useState<string[]>([]);
  const calls = useCalls(q);
  const scenarios = useScenarios();
  const personas = usePersonas();
  const profiles = useProfiles();
  const [sel, setSel] = useState(0);
  const [sorting, setSorting] = useState<SortingState>([]);
  const set = (k: string, v: string | null) => { const n = new URLSearchParams(sp); if (v) n.set(k, v); else n.delete(k); n.delete("cursor"); setSp(n, { replace: true }); setSel(0); setCursorStack([]); };
  const setVerdict = (v: Verdict | "") => { const n = new URLSearchParams(sp); n.delete("verdict"); n.delete("cursor"); if (v) n.set("verdict", v); setSp(n, { replace: true }); setSel(0); setCursorStack([]); };
  const goPage = (cursor: string | null, dir: "next" | "prev") => {
    const n = new URLSearchParams(sp); if (cursor) n.set("cursor", cursor); else n.delete("cursor");
    setCursorStack((st) => (dir === "next" ? [...st, sp.get("cursor") ?? ""] : st.slice(0, -1)));
    setSp(n); setSel(0);
  };

  const items = calls.data?.items ?? [];
  const columns = useMemo(() => [
    col.display({ id: "sel", cell: (c) => <span style={{ color: "var(--ink-4)" }}>{c.row.index === sel ? "▸" : ""}</span>, size: 18 }),
    col.accessor("id", { header: "Call", cell: (c) => <span className="id">{c.getValue()}</span> }),
    col.accessor("verdict", { header: "Verdict", cell: (c) => { const r = c.row.original; return <>{r.status !== "done" ? <VerdictBadge v={r.status as "running" | "queued" | "evaluating" | "failed"} /> : r.verdict ? <VerdictBadge v={r.verdict} /> : null}{r.humanVerdict && r.humanVerdict !== r.verdict ? <div className="faint" style={{ fontSize: 10 }}>human: {verdictLabel[r.humanVerdict]}</div> : null}{r.invalidReason ? <div className="faint" style={{ fontSize: 10, maxWidth: 220, whiteSpace: "normal" }}>{r.invalidReason}</div> : null}</>; } }),
    col.accessor("scenarioTitle", { header: "Scenario", cell: (c) => c.getValue() ? <span>{c.getValue()}</span> : <span className="faint">unknown{c.row.original.mode === "upload" ? " (upload)" : ""}</span> }),
    col.accessor("personaName", { header: "Persona", cell: (c) => c.getValue() ?? <span className="faint">unknown</span> }),
    col.accessor("mode", { header: "Mode", cell: (c) => <ModeBadge mode={c.getValue()} /> }),
    col.accessor("profile", { header: "Profile", cell: (c) => <span className="mono">{c.getValue()} {isUntested(c.row.original, profiles.data?.items) && <Untested />}</span> }),
    col.accessor("batchId", { header: "Run", cell: (c) => <span className="id">{c.getValue()} {c.row.original.rep ? <span className="faint">r{c.row.original.rep}</span> : null}</span> }),
    col.accessor("durationS", { header: () => <span className="num">Dur</span>, cell: (c) => <span className="mono">{fmtClock(c.getValue())}</span>, meta: { num: true } }),
    col.accessor("replyP50S", { header: () => <span title="median bot reply latency">Reply p50</span>, cell: (c) => c.getValue() == null ? <span className="faint" title={c.row.original.timingIsReal ? "not measured" : "Turn-based call, timing not measured"}>n/a</span> : <span className="mono">{c.getValue()!.toFixed(2)} s</span>, meta: { num: true } }),
    col.accessor("overlaps", { header: "Overlaps", cell: (c) => c.getValue() == null ? <span className="faint">n/a</span> : <span className="mono">{c.getValue()}</span>, meta: { num: true } }),
    col.accessor("findings", { header: "Findings", cell: (c) => <span className="mono">{c.getValue()}{c.row.original.openFindings ? <> <span className="tag mono" style={{ color: "var(--review)" }}>{c.row.original.openFindings} open</span></> : null}</span>, meta: { num: true } }),
  ], [sel, profiles.data]);
  const table = useReactTable({ data: items, columns, state: { sorting }, onSortingChange: setSorting, getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel() });
  const rows = table.getRowModel().rows;
  const open = (r: CallSummary) => nav(`/calls/${r.id}`);
  useHotkeys({
    j: () => setSel((x) => Math.min(rows.length - 1, x + 1)), k: () => setSel((x) => Math.max(0, x - 1)),
    Enter: () => rows[sel] && open(rows[sel].original), o: () => rows[sel] && open(rows[sel].original),
    Escape: () => { setSp(new URLSearchParams(), { replace: true }); setSel(0); },
  });
  useEffect(() => { document.querySelector<HTMLElement>(`[data-row="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const facets = calls.data?.facets.verdict ?? {};
  const total = calls.data?.facets.all ?? 0;
  const activeFilters = [...sp.keys()].length;
  return (
    <Shell crumbs={["Calls"]} keys={KEYS}>
      <div className="page">
        <div className="page-head"><div><h1>Calls</h1><div className="page-sub">{calls.data ? `${calls.data.total} of ${total} calls match${calls.data.total > items.length ? `, showing ${items.length} on this page` : ""}.` : ""} Click a row to review it. Invalid calls are test problems and are excluded from pass rates.</div></div>
          <div className="actions"><Button size="sm" onClick={() => exportCsv(items)} disabled={!items.length} title="Exports the rows on this page only">Export CSV ({items.length} on this page)</Button></div></div>
        <div className={s.filters} role="group" aria-label="Filters">
          <span className="seg" role="group" aria-label="Verdict">
            <button type="button" aria-pressed={q.verdict!.length === 0} onClick={() => setVerdict("")}>all <span className="faint num">{total}</span></button>
            {VERDICTS.map((v) => <button key={v} type="button" aria-pressed={q.verdict!.length === 1 && q.verdict![0] === v} onClick={() => setVerdict(v)}>{verdictLabel[v]} <span className="faint num">{facets[v] ?? 0}</span></button>)}
          </span>
          <select className="in" aria-label="Scenario" value={q.scenario ?? ""} onChange={(e) => set("scenario", e.target.value || null)}><option value="">Any scenario</option>{scenarios.data?.items.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}</select>
          <select className="in" aria-label="Persona" value={q.persona ?? ""} onChange={(e) => set("persona", e.target.value || null)}><option value="">Any persona</option>{personas.data?.items.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</select>
          <select className="in" aria-label="Mode" value={q.mode ?? ""} onChange={(e) => set("mode", e.target.value || null)}><option value="">Any mode</option><option value="live_bots">live</option><option value="turn_based">turn-based</option><option value="human">mic</option><option value="upload">upload</option></select>
          <select className="in" aria-label="Model profile" value={q.profile ?? ""} onChange={(e) => set("profile", e.target.value || null)}><option value="">Any profile</option>{profiles.data?.items.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</select>
          {q.run && <button type="button" className="chip" aria-pressed="true" onClick={() => set("run", null)}>{q.run} <span className="x">×</span></button>}
          <span className="grow" />
          <label className="check"><input type="checkbox" checked={!!q.openOnly} onChange={(e) => set("open", e.target.checked ? "1" : null)} /> open findings only</label>
        </div>
        <Panel flush>
          <div style={{ overflow: "auto" }}>
            <QueryState q={calls} what="calls" skeleton={<div style={{ padding: 12 }}><Skeleton rows={6} /></div>}>
              {() => rows.length === 0 ? (
                <Empty title="No calls match these filters">{activeFilters ? <>Clear a filter (<kbd>Esc</kbd>), or </> : "Nothing has run yet. "}<a href="/runs/new" onClick={(e) => { e.preventDefault(); nav("/runs/new"); }}>launch a run</a>{q.scenario ? " for this scenario" : ""}.</Empty>
              ) : (
                <table className="t">
                  <thead>{table.getHeaderGroups().map((hg) => <tr key={hg.id}>{hg.headers.map((h) => <th key={h.id} className={(h.column.columnDef.meta as { num?: boolean } | undefined)?.num ? "num" : ""} onClick={h.column.getToggleSortingHandler()} style={{ cursor: h.column.getCanSort() ? "pointer" : undefined }}>{flexRender(h.column.columnDef.header, h.getContext())}{{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted() as string] ?? ""}</th>)}</tr>)}</thead>
                  <tbody>{rows.map((r) => <tr key={r.id} data-row={r.index} tabIndex={0} className={r.index === sel ? "sel" : ""} onClick={() => open(r.original)} onFocus={() => setSel(r.index)} onKeyDown={(e) => e.key === "Enter" && open(r.original)}>{r.getVisibleCells().map((c) => <td key={c.id} className={(c.column.columnDef.meta as { num?: boolean } | undefined)?.num ? "num" : ""}>{flexRender(c.column.columnDef.cell, c.getContext())}</td>)}</tr>)}</tbody>
                </table>
              )}
            </QueryState>
          </div>
          {calls.data && (calls.data.nextCursor || cursorStack.length > 0) && (
            <div className="ph" style={{ borderTop: "1px solid var(--line-1)", borderBottom: 0 }}>
              <span className="faint">page {cursorStack.length + 1}{calls.data.total ? `, ${calls.data.total} matching calls` : ""}</span><span className="grow" />
              <Button size="sm" disabled={cursorStack.length === 0} onClick={() => goPage(cursorStack[cursorStack.length - 1] || null, "prev")}>Previous page</Button>
              <Button size="sm" disabled={!calls.data.nextCursor} onClick={() => goPage(calls.data!.nextCursor, "next")}>Next page</Button>
            </div>
          )}
        </Panel>
      </div>
    </Shell>
  );
}

function exportCsv(items: CallSummary[]) {
  const head = ["id", "verdict", "humanVerdict", "scenario", "persona", "mode", "profile", "run", "rep", "durationS", "replyP50S", "overlaps", "findings", "openFindings", "startedAt"];
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = items.map((c) => [c.id, c.verdict, c.humanVerdict, c.scenarioTitle, c.personaName, c.mode, c.profile, c.batchId, c.rep, c.durationS, c.replyP50S, c.overlaps, c.findings, c.openFindings, c.startedAt].map(esc).join(","));
  const blob = new Blob([[head.join(","), ...rows].join("\n")], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "calls.csv"; a.click(); URL.revokeObjectURL(a.href);
}
