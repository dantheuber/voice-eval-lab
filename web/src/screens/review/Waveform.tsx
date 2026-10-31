import { useEffect, useRef } from "react";
import { useTheme } from "@/store/theme";

/* Draws server peaks (0..1, perSecond samples) as mirrored bars, with overlap intervals hatched underneath. */
export function Waveform({ peaks, perSecond, durationS, overlaps, color, height = 56, onSeek }: {
  peaks: number[]; perSecond: number; durationS: number; overlaps: [number, number][]; color: "caller" | "bot"; height?: number; onSeek: (t: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useTheme((s) => s.theme);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const draw = () => {
      const w = c.clientWidth; if (!w) return;
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.floor(w * dpr); c.height = Math.floor(height * dpr);
      const ctx = c.getContext("2d"); if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, height);
      const css = getComputedStyle(document.documentElement);
      const fill = css.getPropertyValue(color === "caller" ? "--caller" : "--bot").trim();
      const over = css.getPropertyValue("--overlap").trim();
      for (const [a, b] of overlaps) {
        const x0 = (a / durationS) * w, x1 = (b / durationS) * w;
        ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, height); ctx.clip();
        ctx.strokeStyle = over; ctx.lineWidth = 2;
        for (let x = x0 - height; x < x1 + height; x += 6) { ctx.beginPath(); ctx.moveTo(x, height); ctx.lineTo(x + height, 0); ctx.stroke(); }
        ctx.restore();
      }
      ctx.fillStyle = fill;
      const mid = height / 2;
      const total = peaks.length;
      const samplesPerPx = total / w;
      for (let x = 0; x < w; x += 2) {
        const i0 = Math.floor(x * samplesPerPx), i1 = Math.max(i0 + 1, Math.floor((x + 2) * samplesPerPx));
        let m = 0;
        for (let i = i0; i < i1 && i < total; i++) if (peaks[i] > m) m = peaks[i];
        const bh = Math.max(1, m * (height - 6));
        ctx.fillRect(x, mid - bh / 2, 1.4, bh);
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [peaks, perSecond, durationS, overlaps, color, height, theme]);
  return <canvas ref={ref} height={height} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onSeek(((e.clientX - r.left) / r.width) * durationS); }} role="presentation" />;
}
