/* Single-letter keys for the review loop, with a scope stack so dialogs and inputs do not fire them.
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

let pendingChord: string | null = null;
let chordTimer: ReturnType<typeof setTimeout> | undefined;

export function useHotkeys(map: HotkeyMap, opts: { scope?: string; enabled?: boolean } = {}) {
  const ref = useRef(map);
  ref.current = map;
  const scope = opts.scope ?? "page";
  const enabled = opts.enabled ?? true;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (topScope() !== scope) return;
      const typingAllowed = e.ctrlKey || e.metaKey || e.key === "Escape";
      if (isTyping(e) && !typingAllowed) return;
      const id = keyId(e);
      const m = ref.current;
      if (pendingChord) {
        const chord = `${pendingChord} ${e.key}`;
        pendingChord = null; clearTimeout(chordTimer);
        if (m[chord]) { e.preventDefault(); m[chord](e); return; }
      }
      const chordStarts = Object.keys(m).some((k) => k.startsWith(`${e.key} `));
      if (chordStarts && !m[id]) { pendingChord = e.key; clearTimeout(chordTimer); chordTimer = setTimeout(() => { pendingChord = null; }, 900); return; }
      const fn = m[id];
      if (fn) { e.preventDefault(); fn(e); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [scope, enabled]);
}

export function useScope(name: string, active = true) {
  useEffect(() => { if (!active) return; return pushScope(name); }, [name, active]);
}
