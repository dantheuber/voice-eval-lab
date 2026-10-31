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

export interface Frac { pass: number; n: number }

/** Direction of change between two pass fractions, judged on rates, not counts. Null when either side has no valid calls. */
export function rateDirection(a: Frac, b: Frac): "better" | "worse" | "same" | null {
  if (!a.n || !b.n) return null;
  const ra = a.pass / a.n, rb = b.pass / b.n;
  if (Math.abs(ra - rb) < 1e-9) return "same";
  return rb > ra ? "better" : "worse";
}

/** Whether two Wilson intervals share any range. Null when either sample is empty. */
export function intervalsOverlap(a: Frac, b: Frac): boolean | null {
  const wa = wilson(a.pass, a.n), wb = wilson(b.pass, b.n);
  if (!wa || !wb) return null;
  return wa[0] <= wb[1] && wb[0] <= wa[1];
}

/** Runs async writes for one key one at a time, in order, so a slow earlier write cannot overwrite a later one. */
const queues = new Map<string, Promise<unknown>>();
export function serialize<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const prev = queues.get(key) ?? Promise.resolve();
  const next = prev.catch(() => undefined).then(fn);
  queues.set(key, next);
  next.finally(() => { if (queues.get(key) === next) queues.delete(key); }).catch(() => undefined);
  return next;
}

/** Median of a list, or null. */
export function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
