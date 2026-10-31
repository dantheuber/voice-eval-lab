/* Single-letter keys for the review loop. One window listener dispatches to a registry of maps, so chords
   (`g o`) are owned centrally and a key fires exactly once even when several components register in a scope.
   Key strings: "j", "Shift+ArrowLeft", "Ctrl+Enter", " " (space), "?". Chords: "g o". */
import { useEffect, useRef } from "react";

export type HotkeyMap = Record<string, (e: KeyboardEvent) => void>;

const scopes: string[] = [];
export function pushScope(name: string) { scopes.push(name); return () => { const i = scopes.lastIndexOf(name); if (i >= 0) scopes.splice(i, 1); }; }
const topScope = () => scopes[scopes.length - 1] ?? "page";

function isTyping(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName?.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
}

export function keyId(e: KeyboardEvent): string {
  const mods = [e.ctrlKey || e.metaKey ? "Ctrl" : "", e.shiftKey && e.key.length > 1 ? "Shift" : "", e.altKey ? "Alt" : ""].filter(Boolean);
  return [...mods, e.key].join("+");
}

interface Entry { scope: string; map: React.RefObject<HotkeyMap>; order: number }
const registry = new Set<Entry>();
let order = 0;
let pendingChord: string | null = null;
let chordTimer: ReturnType<typeof setTimeout> | undefined;
let installed = false;

export function dispatchKey(e: KeyboardEvent): boolean {
  const scope = topScope();
  const active = [...registry].filter((r) => r.scope === scope).sort((a, b) => b.order - a.order);
  if (!active.length) return false;
  const typingAllowed = e.ctrlKey || e.metaKey || e.key === "Escape";
  if (isTyping(e) && !typingAllowed) return false;
  const id = keyId(e);
  const find = (k: string) => { for (const r of active) { const fn = r.map.current?.[k]; if (fn) return fn; } return null; };
  if (pendingChord) {
    const chord = `${pendingChord} ${e.key}`;
    pendingChord = null; clearTimeout(chordTimer);
    const fn = find(chord);
    if (fn) { e.preventDefault(); fn(e); return true; }
  }
  const direct = find(id);
  const startsChord = !direct && active.some((r) => Object.keys(r.map.current ?? {}).some((k) => k.startsWith(`${e.key} `)));
  if (startsChord) { pendingChord = e.key; clearTimeout(chordTimer); chordTimer = setTimeout(() => { pendingChord = null; }, 900); return true; }
  if (direct) { e.preventDefault(); direct(e); return true; }
  return false;
}

function install() {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("keydown", dispatchKey);
}

export function useHotkeys(map: HotkeyMap, opts: { scope?: string; enabled?: boolean } = {}) {
  const ref = useRef(map);
  ref.current = map;
  const scope = opts.scope ?? "page";
  const enabled = opts.enabled ?? true;
  useEffect(() => {
    if (!enabled) return;
    install();
    const entry: Entry = { scope, map: ref, order: order++ };
    registry.add(entry);
    return () => { registry.delete(entry); };
  }, [scope, enabled]);
}

export function useScope(name: string, active = true) {
  useEffect(() => { if (!active) return; return pushScope(name); }, [name, active]);
}

/** Test seam: clear chord state between tests. */
export function _resetHotkeys() { pendingChord = null; clearTimeout(chordTimer); }
