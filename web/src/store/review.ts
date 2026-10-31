/* Review session state shared by the timeline, transcript, and findings panel. */
import { create } from "zustand";

export type Clip = [number, number] | null;

interface ReviewState {
  callId: string | null;
  t: number;
  playing: boolean;
  speed: number;
  selected: string | null;
  clip: Clip;
  filter: "all" | "findings" | "events";
  noteOpen: string | null;
  mute: { caller: boolean; bot: boolean };
  seekRequest: number | null; // set by UI, consumed by the audio element
  setCall: (id: string) => void;
  setT: (t: number) => void;
  setPlaying: (p: boolean) => void;
  cycleSpeed: () => void;
  select: (id: string | null) => void;
  setClip: (c: Clip) => void;
  setFilter: (f: ReviewState["filter"]) => void;
  setNoteOpen: (id: string | null) => void;
  toggleMute: (ch: "caller" | "bot") => void;
  seek: (t: number) => void;
  consumeSeek: () => void;
}

export const useReview = create<ReviewState>((set, get) => ({
  callId: null, t: 0, playing: false, speed: 1, selected: null, clip: null, filter: "all", noteOpen: null, mute: { caller: false, bot: false }, seekRequest: null,
  setCall: (id) => { if (get().callId !== id) set({ callId: id, t: 0, playing: false, selected: null, clip: null, noteOpen: null, seekRequest: null }); },
  setT: (t) => set({ t }),
  setPlaying: (playing) => set({ playing }),
  cycleSpeed: () => set((s) => ({ speed: s.speed === 1 ? 1.5 : s.speed === 1.5 ? 2 : 1 })),
  select: (selected) => set({ selected }),
  setClip: (clip) => set({ clip }),
  setFilter: (filter) => set({ filter }),
  setNoteOpen: (noteOpen) => set({ noteOpen }),
  toggleMute: (ch) => set((s) => ({ mute: { ...s.mute, [ch]: !s.mute[ch] } })),
  seek: (t) => set({ t: Math.max(0, t), seekRequest: Math.max(0, t) }),
  consumeSeek: () => set({ seekRequest: null }),
}));
