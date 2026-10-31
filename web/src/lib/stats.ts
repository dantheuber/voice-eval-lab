import type { Counts } from "@/api/types";

/** Wilson score interval, 95%, as [low, high] in 0..1. Returns null when n is 0. */
export function wilson(k: number, n: number, z = 1.96): [number, number] | null {
  if (n <= 0) return null;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n);
  const m = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, (c - m) / d), Math.min(1, (c + m) / d)];
}

export function ciText(k: number, n: number): string {
  const w = wilson(k, n);
  if (!w) return "";
  return `${Math.round(w[0] * 100)}-${Math.round(w[1] * 100)}%`;
}

export function valid(c: Counts): number {
  return c.pass + c.fail + c.needsReview;
}

export function passPct(c: Counts): number | null {
  const v = valid(c);
  return v ? Math.round((100 * c.pass) / v) : null;
}

export function addCounts(a: Counts, b: Counts): Counts {
  return { pass: a.pass + b.pass, fail: a.fail + b.fail, needsReview: a.needsReview + b.needsReview, invalid: a.invalid + b.invalid };
}

export const zeroCounts = (): Counts => ({ pass: 0, fail: 0, needsReview: 0, invalid: 0 });

/** Median of a list, or null. */
export function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
