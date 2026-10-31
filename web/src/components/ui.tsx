/* Small primitives styled by the global classes in styles/base.css (ported from the prototype). */
import type { ButtonHTMLAttributes, ReactNode } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import type { CallMode, Counts, HealthState, Verdict } from "@/api/types";
import { modeLabel, modeTitle, verdictLabel } from "@/lib/format";
import { ciText, passPct, valid } from "@/lib/stats";
import { ApiError } from "@/api/transport";

const PATHS: Record<string, string> = {
  overview: '<path d="M2 9.5h4V14H2zM6 2h4v12H6zM10 6h4v8h-4z"/>',
  calls: '<path d="M2 3h12M2 8h12M2 13h8"/>',
  compare: '<path d="M2 3h5v10H2zM9 3h5v10H9z"/>',
  scenario: '<path d="M3 2h7l3 3v9H3z"/><path d="M6 8h4M6 11h4"/>',
  persona: '<circle cx="8" cy="5" r="3"/><path d="M2 14c1-3 3-4 6-4s5 1 6 4"/>',
  run: '<path d="M4 2l10 6-10 6z"/>',
  talk: '<rect x="5.5" y="1.5" width="5" height="8" rx="2.5"/><path d="M3 7.5a5 5 0 0 0 10 0M8 12.5v2"/>',
  upload: '<path d="M8 11V3M4.5 6.5 8 3l3.5 3.5M2 13h12"/>',
  settings: '<circle cx="8" cy="8" r="2.2"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4"/>',
  play: '<path d="M4 2.5l9 5.5-9 5.5z"/>',
  pause: '<path d="M4.5 3v10M11.5 3v10"/>',
  check: '<path d="M3 8.5l3 3 7-7"/>',
  x: '<path d="M4 4l8 8M12 4l-8 8"/>',
  warn: '<path d="M8 2 1.5 13.5h13z"/><path d="M8 6.5v3.5M8 12v.5"/>',
  info: '<circle cx="8" cy="8" r="6.5"/><path d="M8 7v4M8 5v.5"/>',
  sun: '<circle cx="8" cy="8" r="3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M3 13l1.4-1.4M11.6 4.4 13 3"/>',
  moon: '<path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z"/>',
  live: '<path d="M1 8h3l2-4 3 8 2-5 1 1h3"/>',
  turn: '<path d="M3 5h8l-2-2M13 11H5l2 2"/>',
  headphones: '<path d="M2 10V8a6 6 0 0 1 12 0v2"/><rect x="2" y="9" width="3" height="5" rx="1"/><rect x="11" y="9" width="3" height="5" rx="1"/>',
  tool: '<path d="M9.5 2.5a3.5 3.5 0 0 0 4 4L8 12l-2 2-2-2 2-2 5.5-5.5"/>',
  db: '<ellipse cx="8" cy="4" rx="5.5" ry="2"/><path d="M2.5 4v8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2"/>',
  volume: '<path d="M2 6h3l4-3v10l-4-3H2z"/><path d="M11 6a3 3 0 0 1 0 4"/>',
  mute: '<path d="M2 6h3l4-3v10l-4-3H2z"/><path d="M11 6l3 4M14 6l-3 4"/>',
};
export function Icon({ name, size = 14, className = "" }: { name: keyof typeof PATHS | string; size?: number; className?: string }) {
  return <svg className={`ico ${className}`} width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" dangerouslySetInnerHTML={{ __html: PATHS[name] ?? "" }} />;
}

export function Kbd({ children }: { children: ReactNode }) { return <kbd>{children}</kbd>; }

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "danger" | "dangerSolid" | "ghost"; size?: "sm" | "md" | "lg"; tone?: "pass" | "fail" };
export function Button({ variant = "default", size = "md", tone, className = "", children, ...rest }: BtnProps) {
  const v = variant === "primary" ? "primary" : variant === "danger" ? "danger" : variant === "dangerSolid" ? "danger solid" : variant === "ghost" ? "ghost" : "";
  return <button type="button" className={`btn ${v} ${size === "md" ? "" : size} ${tone ?? ""} ${className}`} {...rest}>{children}</button>;
}

export function VerdictBadge({ v, label, title }: { v: Verdict | "running" | "queued" | "evaluating" | "failed"; label?: string; title?: string }) {
  return <span className={`v ${v}`} title={title}>{label ?? verdictLabel[v]}</span>;
}

export function ModeBadge({ mode }: { mode: CallMode }) {
  const cls = mode === "turn_based" ? "turn" : mode;
  const ico = mode === "live_bots" ? "live" : mode === "turn_based" ? "turn" : mode === "human" ? "talk" : "upload";
  return <span className={`mode ${cls}`} title={modeTitle[mode]}><Icon name={ico} size={11} />{modeLabel[mode]}</span>;
}

export function HealthDot({ state, children, title }: { state: HealthState; children?: ReactNode; title?: string }) {
  return <span className={`health ${state}`} title={title}>{children}</span>;
}

export function Note({ kind = "plain", children, style }: { kind?: "plain" | "info" | "warn" | "err"; children: ReactNode; style?: React.CSSProperties }) {
  const ico = kind === "warn" ? "warn" : kind === "err" ? "warn" : "info";
  return <div className={`note ${kind === "plain" ? "" : kind}`} role={kind === "err" ? "alert" : undefined} style={style}><Icon name={ico} /><span>{children}</span></div>;
}

export function Panel({ title, meta, actions, children, flush, className = "", style }: { title?: ReactNode; meta?: ReactNode; actions?: ReactNode; children: ReactNode; flush?: boolean; className?: string; style?: React.CSSProperties }) {
  return (
    <div className={`panel ${flush ? "flush" : ""} ${className}`} style={style}>
      {(title || actions) && <div className="ph"><h2>{title}</h2>{meta && <span className="faint">{meta}</span>}<span className="grow" />{actions}</div>}
      <div className="pb">{children}</div>
    </div>
  );
}

export function RateBar({ counts, title }: { counts: Counts; title?: string }) {
  const n = counts.pass + counts.fail + counts.needsReview + counts.invalid;
  const seg = (c: string, v: number, lab: string) => (v ? <i key={c} className={c} style={{ flex: v }} title={`${lab}: ${v}`} /> : null);
  return (
    <div className="rate" role="img" title={title} aria-label={`${counts.pass} pass, ${counts.fail} fail, ${counts.needsReview} needs review, ${counts.invalid} invalid of ${n}`}>
      {seg("p", counts.pass, "pass")}{seg("f", counts.fail, "fail")}{seg("r", counts.needsReview, "needs review")}{seg("i", counts.invalid, "invalid")}
    </div>
  );
}
export function RateLegend({ counts }: { counts?: Counts }) {
  const c = counts;
  return (
    <div className="rate-leg">
      <span style={{ "--c": "var(--pass)" } as React.CSSProperties}>pass{c ? ` ${c.pass}` : ""}</span>
      <span style={{ "--c": "var(--fail)" } as React.CSSProperties}>fail{c ? ` ${c.fail}` : ""}</span>
      <span style={{ "--c": "var(--review)" } as React.CSSProperties}>needs review{c ? ` ${c.needsReview}` : ""}</span>
      <span style={{ "--c": "var(--invalid)" } as React.CSSProperties}>invalid{c ? ` ${c.invalid}` : ""} (not counted)</span>
    </div>
  );
}
export function PassRate({ counts, bold = true }: { counts: Counts; bold?: boolean }) {
  const p = passPct(counts);
  if (p == null) return <span className="faint">no valid calls</span>;
  return <>{bold ? <b>{p}%</b> : <span>{p}%</span>} <span className="ci">{ciText(counts.pass, valid(counts))}</span></>;
}

export function Field({ label, help, children, style, className = "" }: { label: ReactNode; help?: ReactNode; children: ReactNode; style?: React.CSSProperties; className?: string }) {
  return <div className={`field ${className}`} style={style}><label>{label}</label>{children}{help && <span className="help">{help}</span>}</div>;
}

export function Seg<V extends string>({ value, options, onChange, ariaLabel }: { value: V; options: { value: V; label: ReactNode; title?: string }[]; onChange: (v: V) => void; ariaLabel?: string }) {
  return (
    <span className="seg" role="group" aria-label={ariaLabel}>
      {options.map((o) => <button key={o.value} type="button" title={o.title} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>)}
    </span>
  );
}

export function Scale({ label, value, onChange, low, high }: { label: string; value: number; onChange: (v: number) => void; low: string; high: string }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="scale" role="radiogroup" aria-label={label}>
        {[1, 2, 3, 4, 5].map((n) => <button key={n} type="button" role="radio" aria-checked={n === value} aria-pressed={n === value} onClick={() => onChange(n)}>{n}</button>)}
      </div>
      <div className="scale-cap"><span>{low}</span><span>{high}</span></div>
    </div>
  );
}

export function Skeleton({ rows = 3, width }: { rows?: number; width?: string[] }) {
  return <div aria-busy="true" aria-label="Loading" style={{ display: "flex", flexDirection: "column", gap: 8 }}>{Array.from({ length: rows }, (_, i) => <div key={i} className="skel" style={{ width: width?.[i % (width.length || 1)] ?? `${90 - (i % 3) * 20}%` }} />)}</div>;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><b>{title}</b>{children}</div>;
}

export function ErrorBox({ error, retry, what }: { error: unknown; retry?: () => void; what?: string }) {
  const e = error as Partial<ApiError> & { message?: string };
  if (e instanceof ApiError && e.code === "model_not_ready") {
    const d = e.detail as { role?: string; model?: string; elapsedS?: number; usualS?: number } | undefined;
    return <Note kind="warn"><b>{d?.role ?? "A model"} is loading</b>{d?.model ? ` (${d.model})` : ""}{d?.elapsedS != null ? `, ${d.elapsedS} s so far` : ""}{d?.usualS != null ? `, usually under ${d.usualS} s` : ""}. {retry && <a href="#" onClick={(ev) => { ev.preventDefault(); retry(); }}>Try again</a>}</Note>;
  }
  return <Note kind="err"><b>Could not load{what ? ` ${what}` : ""}.</b> {e?.message ?? String(error)} {retry && <a href="#" onClick={(ev) => { ev.preventDefault(); retry(); }}>Retry</a>}</Note>;
}

/* Renders loading, error, or children for a query result. */
export function QueryState<D>({ q, children, skeleton, what }: { q: { data: D | undefined; isPending: boolean; isError: boolean; error: unknown; refetch: () => unknown }; children: (d: D) => ReactNode; skeleton?: ReactNode; what?: string }) {
  if (q.isPending) return <>{skeleton ?? <Skeleton />}</>;
  if (q.isError) return <ErrorBox error={q.error} retry={() => q.refetch()} what={what} />;
  return <>{children(q.data as D)}</>;
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={300}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal><Tooltip.Content className="tooltip" sideOffset={4}>{content}</Tooltip.Content></Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Untested() {
  return <span className="untested" title="Unit tested against mocks only. No Bedrock access on this machine."><Icon name="warn" size={11} /> untested against live AWS</span>;
}
