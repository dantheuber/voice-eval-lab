import { create } from "zustand";

type Theme = "dark" | "light";

function initial(): Theme {
  const el = document.documentElement.dataset.theme;
  return el === "light" ? "light" : "dark";
}

export const useTheme = create<{ theme: Theme; toggle: () => void; set: (t: Theme) => void }>((set) => ({
  theme: initial(),
  set: (t) => {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("vl-theme", t); } catch { /* private mode */ }
    set({ theme: t });
  },
  toggle: () => {
    const t: Theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("vl-theme", t); } catch { /* private mode */ }
    set({ theme: t });
  },
}));
