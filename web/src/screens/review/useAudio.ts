/* Drives an <audio> element from the review store and routes it through Web Audio so either channel can be muted.
   Caller is the left channel, bot the right, matching the recorder. */
import { useEffect, useRef } from "react";
import { useReview } from "@/store/review";
import { toast } from "@/store/toast";

export function useAudio(src: string | null, durationS: number) {
  const el = useRef<HTMLAudioElement | null>(null);
  const graph = useRef<{ ctx: AudioContext; gains: [GainNode, GainNode] } | null>(null);
  const clipEnd = useRef<number | null>(null);

  // element lifecycle
  useEffect(() => {
    if (!src) return;
    const a = new Audio();
    a.preload = "auto";
    a.crossOrigin = "anonymous";
    a.src = src;
    el.current = a;
    const st = useReview.getState();
    a.playbackRate = st.speed;
    const onTime = () => {
      const s = useReview.getState();
      s.setT(a.currentTime);
      if (clipEnd.current != null && a.currentTime >= clipEnd.current) { a.pause(); clipEnd.current = null; }
    };
    const onPlay = () => useReview.getState().setPlaying(true);
    const onPause = () => useReview.getState().setPlaying(false);
    const onEnded = () => { useReview.getState().setPlaying(false); };
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnded);
    // smoother playhead than timeupdate's ~250 ms
    let raf = 0;
    const tick = () => { if (!a.paused) { useReview.getState().setT(a.currentTime); } raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      a.pause();
      a.removeEventListener("timeupdate", onTime); a.removeEventListener("play", onPlay); a.removeEventListener("pause", onPause); a.removeEventListener("ended", onEnded);
      a.src = ""; el.current = null;
      graph.current?.ctx.close().catch(() => undefined); graph.current = null;
    };
  }, [src]);

  const ensureGraph = () => {
    const a = el.current;
    if (!a || graph.current) return graph.current;
    try {
      const ctx = new AudioContext();
      const source = ctx.createMediaElementSource(a);
      const split = ctx.createChannelSplitter(2);
      const merge = ctx.createChannelMerger(2);
      const gL = ctx.createGain(), gR = ctx.createGain();
      source.connect(split);
      split.connect(gL, 0); split.connect(gR, 1);
      gL.connect(merge, 0, 0); gR.connect(merge, 0, 1);
      merge.connect(ctx.destination);
      graph.current = { ctx, gains: [gL, gR] };
      applyMute();
    } catch { /* no Web Audio (tests); plain element playback still works */ }
    return graph.current;
  };
  const applyMute = () => {
    const g = graph.current; if (!g) return;
    const m = useReview.getState().mute;
    g.gains[0].gain.value = m.caller ? 0 : 1;
    g.gains[1].gain.value = m.bot ? 0 : 1;
  };

  // subscribe to store intents: seek, play/pause, speed, mute
  useEffect(() => {
    const unsub = useReview.subscribe((s, prev) => {
      const a = el.current; if (!a) return;
      if (s.seekRequest != null && s.seekRequest !== prev.seekRequest) {
        a.currentTime = Math.min(durationS, s.seekRequest);
        clipEnd.current = s.clip ? s.clip[1] : null;
        s.consumeSeek();
      }
      if (s.speed !== prev.speed) a.playbackRate = s.speed;
      if (s.mute !== prev.mute) applyMute();
    });
    return unsub;
  }, [durationS]);

  const controls = {
    play: async () => {
      const a = el.current; if (!a) return;
      const g = ensureGraph(); if (g && g.ctx.state === "suspended") await g.ctx.resume().catch(() => undefined);
      const s = useReview.getState();
      clipEnd.current = s.clip && a.currentTime < s.clip[1] ? s.clip[1] : null;
      await a.play().catch((e: DOMException) => {
        if (e?.name === "NotAllowedError") toast("The browser blocked audio until you click somewhere on the page.", 3000);
        else if (e?.name !== "AbortError") toast(`Could not play: ${e?.message ?? e}`, 3000);
      });
    },
    pause: () => el.current?.pause(),
    toggle: async () => { const a = el.current; if (!a) return; if (a.paused) await controls.play(); else a.pause(); },
    playRange: async (start: number, end: number) => {
      const a = el.current; if (!a) return;
      useReview.getState().setClip([start, end]);
      a.currentTime = start; clipEnd.current = end;
      useReview.getState().setT(start);
      await controls.play();
    },
    ready: () => !!el.current,
  };
  return controls;
}
