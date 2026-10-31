/* Synthesises a stereo WAV from speech regions so <audio> playback and channel muting work with no backend.
   Caller is on the left, bot on the right, matching the recorder. */

type Seg = [number, number];

export function synthStereoWav(caller: Seg[], bot: Seg[], durationS: number, sampleRate = 8000): Blob {
  const n = Math.ceil(durationS * sampleRate);
  const buf = new ArrayBuffer(44 + n * 4);
  const dv = new DataView(buf);
  const w = (o: number, s: string) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); dv.setUint32(4, 36 + n * 4, true); w(8, "WAVE"); w(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 2, true); dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true); dv.setUint16(32, 4, true); dv.setUint16(34, 16, true); w(36, "data"); dv.setUint32(40, n * 4, true);

  const voice = (segs: Seg[], f0: number, seed: number) => {
    const out = new Float32Array(n);
    for (const [a, b] of segs) {
      const s0 = Math.floor(a * sampleRate), s1 = Math.min(n, Math.floor(b * sampleRate));
      for (let i = s0; i < s1; i++) {
        const t = i / sampleRate;
        const rel = (t - a) / (b - a);
        const env = Math.min(1, rel * 10, (1 - rel) * 8);
        // syllable-like amplitude modulation and a slowly wandering pitch
        const syl = 0.55 + 0.45 * Math.max(0, Math.sin(t * 2 * Math.PI * (3.2 + Math.sin(t * 0.7 + seed))));
        const f = f0 * (1 + 0.08 * Math.sin(t * 1.3 + seed) + 0.03 * Math.sin(t * 5.1));
        const ph = 2 * Math.PI * f * t;
        const v = Math.sin(ph) * 0.6 + Math.sin(2 * ph) * 0.25 + Math.sin(3 * ph) * 0.12 + (Math.random() - 0.5) * 0.08;
        out[i] = v * env * syl * 0.35;
      }
    }
    return out;
  };
  const L = voice(caller, 145, 1.7), R = voice(bot, 205, 4.2);
  let o = 44;
  for (let i = 0; i < n; i++) {
    dv.setInt16(o, Math.max(-1, Math.min(1, L[i])) * 32767, true); o += 2;
    dv.setInt16(o, Math.max(-1, Math.min(1, R[i])) * 32767, true); o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}
