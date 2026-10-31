/* Local microphone for the level meter and device list. Constraints match the brief: echo cancellation on,
   noise suppression on, automatic gain off (gain hides the talk-over the evaluators need to see). */
import { useEffect, useRef, useState } from "react";

export const MIC_CONSTRAINTS: MediaTrackConstraints = { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 };

export function useMic(deviceId: string | null, enabled: boolean) {
  const [level, setLevel] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [outputs, setOutputs] = useState<MediaDeviceInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let raf = 0;
    let s: MediaStream | null = null;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser has no microphone API. Use HTTPS or localhost.");
        s = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { ...MIC_CONSTRAINTS, deviceId: { exact: deviceId } } : MIC_CONSTRAINTS });
        if (!alive) { s.getTracks().forEach((t) => t.stop()); return; }
        setStream(s); setError(null);
        const all = await navigator.mediaDevices.enumerateDevices();
        setDevices(all.filter((d) => d.kind === "audioinput"));
        setOutputs(all.filter((d) => d.kind === "audiooutput"));
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        const src = ctx.createMediaStreamSource(s);
        const an = ctx.createAnalyser(); an.fftSize = 512;
        src.connect(an);
        const buf = new Uint8Array(an.fftSize);
        const tick = () => {
          an.getByteTimeDomainData(buf);
          let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
          const rms = Math.sqrt(sum / buf.length);
          setLevel((prev) => Math.max(rms * 3, prev * 0.85));
          raf = requestAnimationFrame(tick);
        };
        tick();
      } catch (e) {
        const err = e as DOMException;
        setError(err.name === "NotAllowedError" ? "Microphone permission denied. Allow it in the browser's site settings. On the LAN the page must be HTTPS." : err.name === "NotFoundError" ? "No microphone found." : err.message || String(e));
      }
    })();
    return () => {
      alive = false; cancelAnimationFrame(raf);
      s?.getTracks().forEach((t) => t.stop()); setStream(null);
      ctxRef.current?.close().catch(() => undefined); ctxRef.current = null;
    };
  }, [deviceId, enabled]);

  const headphonesLikely = outputs.some((o) => /headphone|headset|earbud|airpods|buds/i.test(o.label));
  return { level, devices, outputs, error, stream, headphonesLikely };
}
