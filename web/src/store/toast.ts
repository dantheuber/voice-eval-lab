import { create } from "zustand";

let timer: ReturnType<typeof setTimeout> | undefined;

export const useToast = create<{ message: string | null; show: (m: string, ms?: number) => void; clear: () => void }>((set) => ({
  message: null,
  show: (message, ms = 1800) => {
    clearTimeout(timer);
    set({ message });
    timer = setTimeout(() => set({ message: null }), ms);
  },
  clear: () => { clearTimeout(timer); set({ message: null }); },
}));

export const toast = (m: string, ms?: number) => useToast.getState().show(m, ms);
