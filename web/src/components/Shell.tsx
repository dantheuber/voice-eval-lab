import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import * as Dialog from "@radix-ui/react-dialog";
import { Button, HealthDot, Icon, Kbd } from "./ui";
import { useTheme } from "@/store/theme";
import { useToast } from "@/store/toast";
import { useHealth, useOverview, useProfiles, useScenarios, useRuns, usePersonas, useSystem } from "@/api/queries";
import { endpoints } from "@/api/client";
import { useHotkeys, useScope } from "@/hooks/useHotkeys";
import { isMock } from "@/api/transport";

const NAV = [
  { group: "Results", items: [{ to: "/", key: "overview", label: "Overview", end: true }, { to: "/calls", key: "calls", label: "Calls" }, { to: "/compare", key: "compare", label: "Compare" }] },
  { group: "Test design", items: [{ to: "/scenarios", key: "scenario", label: "Scenarios" }, { to: "/personas", key: "persona", label: "Personas" }] },
  { group: "Run", items: [{ to: "/runs/new", key: "run", label: "Run launcher" }, { to: "/talk", key: "talk", label: "Talk to the bot" }, { to: "/upload", key: "upload", label: "Upload recording" }] },
  { group: "System", items: [{ to: "/settings", key: "settings", label: "Models and settings" }] },
];

export const GLOBAL_KEYS: [string, string][] = [["Toggle this help", "?"], ["Jump to anything", "Ctrl K"], ["Overview", "g o"], ["Calls", "g c"], ["Next call needing review", "g r"], ["Talk to the bot", "g t"], ["Scenarios", "g s"], ["Run launcher", "g l"]];

export function Shell({ children, crumbs, actions, keys }: { children: ReactNode; crumbs: ReactNode[]; actions?: ReactNode; keys?: [string, string][] }) {
  const { theme, toggle } = useTheme();
  const health = useHealth();
  const overview = useOverview();
  const profiles = useProfiles();
  const system = useSystem();
  const nav = useNavigate();
  const [keysOpen, setKeysOpen] = useState(false);
  const [jumpOpen, setJumpOpen] = useState(false);
  const toast = useToast((s) => s.message);

  const goNextReview = async () => {
    const r = await endpoints.nextReview();
    if (r.callId) nav(`/calls/${r.callId}`); else useToast.getState().show("Nothing needs review");
  };
  useHotkeys({
    "?": () => setKeysOpen((o) => !o),
    "Ctrl+k": () => setJumpOpen(true),
    "g o": () => nav("/"), "g c": () => nav("/calls"), "g r": () => void goNextReview(), "g t": () => nav("/talk"), "g s": () => nav("/scenarios"), "g l": () => nav("/runs/new"),
  }, { scope: "page" });
  // Also fire global keys from screen scopes (review) by listening separately for '?' and Ctrl+K
  useHotkeys({ "?": () => setKeysOpen((o) => !o), "Ctrl+k": () => setJumpOpen(true), "g o": () => nav("/"), "g c": () => nav("/calls"), "g r": () => void goNextReview(), "g t": () => nav("/talk") }, { scope: "review" });

  const items = health.data?.items ?? [];
  const okCount = items.filter((h) => h.state === "ok" || h.state === "off").length;
  const loading = items.find((h) => h.state === "loading");
  const down = items.find((h) => h.state === "down");
  const reviewCount = overview.data?.queueTotal;
  const activeProfile = profiles.data?.activeId ?? "";
  const lanHost = system.data?.urls.lan?.replace(/^https?:\/\//, "") ?? "";

  return (
    <div className="app">
      <nav className="rail" aria-label="Main">
        <div className="brand"><span className="mark" /><div><b>voice-eval-lab</b><small>Fernhill Internet bot{isMock() ? ", fixtures" : ""}</small></div></div>
        {NAV.map((g) => (
          <div key={g.group}>
            <div className="group">{g.group}</div>
            {g.items.map((it) => (
              <NavLink key={it.to} to={it.to} end={it.end} className={({ isActive }) => `item ${isActive ? "cur" : ""}`} aria-current={undefined}>
                {({ isActive }) => (<span className="item-inner" data-active={isActive}><Icon name={it.key} /><span>{it.label}</span>{it.key === "calls" && reviewCount ? <span className="count hot" title={`${reviewCount} calls need review`}>{reviewCount}</span> : null}</span>)}
              </NavLink>
            ))}
          </div>
        ))}
        <div className="spacer" />
        <div className="foot">
          <div className="row"><span>profile</span><span className="mono">{activeProfile || "…"}</span></div>
          <div className="row">
            <Link to="/settings" style={{ color: "inherit" }}>
              <HealthDot state={down ? "down" : loading ? "loading" : items.length ? "ok" : "off"} title={loading ? `${loading.name} loading` : down ? `${down.name} down` : "all models ready"}>{items.length ? `${okCount} of ${items.length} models` : "models"}</HealthDot>
            </Link>
            <Button variant="ghost" size="sm" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={toggle}><Icon name={theme === "dark" ? "sun" : "moon"} /></Button>
          </div>
          <div className="row"><span title="Address for other devices on your network">{lanHost}</span><span><Kbd>?</Kbd> keys</span></div>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <div className="crumbs">{crumbs.map((c, i) => <span key={i} className={i === crumbs.length - 1 ? "cur" : ""}>{c}{i < crumbs.length - 1 && <span className="sep"> / </span>}</span>)}</div>
          <div className="grow" />
          <div className="search" role="button" tabIndex={0} aria-label="Jump to call, scenario, or run" onClick={() => setJumpOpen(true)} onKeyDown={(e) => e.key === "Enter" && setJumpOpen(true)}>Jump to call, scenario, run <Kbd>Ctrl</Kbd><Kbd>K</Kbd></div>
          {actions}
        </header>
        <div className="content">{children}</div>
      </div>
      <KeysDialog open={keysOpen} onOpenChange={setKeysOpen} keys={keys} />
      <JumpDialog open={jumpOpen} onOpenChange={setJumpOpen} />
      <div className={`toast ${toast ? "show" : ""}`} role="status" aria-live="polite">{toast}</div>
    </div>
  );
}

function KeysDialog({ open, onOpenChange, keys }: { open: boolean; onOpenChange: (o: boolean) => void; keys?: [string, string][] }) {
  useScope("dialog", open);
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="overlay show" />
        <Dialog.Content className="dialog dialog-pos" aria-describedby={undefined}>
          <div className="ph"><Dialog.Title asChild><h2>Keyboard</h2></Dialog.Title><span className="grow" /><Dialog.Close asChild><Button variant="ghost" size="sm">Close <Kbd>Esc</Kbd></Button></Dialog.Close></div>
          <div className="pb">
            {keys && <><h3 style={{ marginBottom: 8 }}>This screen</h3><dl className="keys">{keys.map(([a, k]) => <dt key={a}><span>{a}</span><span className="kk">{k.split(" ").map((x, i) => <Kbd key={i}>{x}</Kbd>)}</span></dt>)}</dl></>}
            <h3 style={{ margin: "12px 0 8px" }}>Everywhere</h3>
            <dl className="keys">{GLOBAL_KEYS.map(([a, k]) => <dt key={a}><span>{a}</span><span className="kk">{k.split(" ").map((x, i) => <Kbd key={i}>{x}</Kbd>)}</span></dt>)}</dl>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function JumpDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  useScope("dialog", open);
  const [q, setQ] = useState("");
  const nav = useNavigate();
  const scenarios = useScenarios();
  const personas = usePersonas();
  const runs = useRuns();
  const overview = useOverview();
  useEffect(() => { if (!open) setQ(""); }, [open]);
  const results = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const out: { label: string; sub: string; to: string }[] = [];
    if (needle.startsWith("call_") || /^[0-9a-f]{6,}$/.test(needle)) out.push({ label: needle, sub: "open call", to: `/calls/${needle}` });
    (overview.data?.queue ?? []).forEach((c) => { if (!needle || c.callId.includes(needle) || (c.scenarioTitle ?? "").toLowerCase().includes(needle)) out.push({ label: c.callId, sub: `${c.scenarioTitle ?? "unknown"}, needs review`, to: `/calls/${c.callId}` }); });
    (scenarios.data?.items ?? []).forEach((s) => { if (!needle || s.title.toLowerCase().includes(needle) || s.id.includes(needle)) out.push({ label: s.title, sub: s.id, to: `/scenarios/${s.id}` }); });
    (personas.data?.items ?? []).forEach((p) => { if (!needle || p.name.toLowerCase().includes(needle) || p.id.includes(needle)) out.push({ label: p.name, sub: p.id, to: `/personas/${p.id}` }); });
    (runs.data?.items ?? []).forEach((r) => { if (!needle || r.id.includes(needle) || r.label.toLowerCase().includes(needle)) out.push({ label: r.id, sub: r.label, to: `/runs/${r.id}` }); });
    return out.slice(0, 12);
  }, [q, scenarios.data, personas.data, runs.data, overview.data]);
  const [sel, setSel] = useState(0);
  useEffect(() => setSel(0), [q]);
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="overlay show" />
        <Dialog.Content className="dialog dialog-pos" aria-describedby={undefined}>
          <Dialog.Title className="sr-only">Jump to</Dialog.Title>
          <div className="pb" style={{ paddingBottom: 8 }}>
            <input className="in" autoFocus placeholder="Call id, scenario, persona, run" value={q} onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => { if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(results.length - 1, s + 1)); } if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); } if (e.key === "Enter" && results[sel]) { nav(results[sel].to); onOpenChange(false); } }} />
          </div>
          <div style={{ maxHeight: 360, overflow: "auto", borderTop: "1px solid var(--line-1)" }}>
            {results.length === 0 && <div className="empty">No matches</div>}
            {results.map((r, i) => <div key={r.to + i} className={`jump-row ${i === sel ? "sel" : ""}`} onMouseEnter={() => setSel(i)} onClick={() => { nav(r.to); onOpenChange(false); }}><span>{r.label}</span><span className="faint mono">{r.sub}</span></div>)}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
