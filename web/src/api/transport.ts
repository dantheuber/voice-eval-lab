/* One seam for all network access. The real transport talks to /api and /ws on the same origin.
   The mock transport (VITE_API_MOCK=1 or ?mock=1) answers in-browser from Fernhill fixtures. */

import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  code: string;
  status: number;
  detail: Record<string, unknown> | undefined;
  constructor(status: number, body: ApiErrorBody["error"] | null, fallback: string) {
    super(body?.message ?? fallback);
    this.name = "ApiError";
    this.status = status;
    this.code = body?.code ?? (status === 404 ? "not_found" : status >= 500 ? "internal" : "invalid");
    this.detail = body?.detail;
  }
}

export type Method = "GET" | "POST" | "PUT" | "DELETE";

/* A minimal socket shape both WebSocket and the mock satisfy. */
export interface Socket {
  send(data: string): void;
  close(): void;
  onmessage: ((data: unknown) => void) | null;
  onclose: (() => void) | null;
  onerror: ((err: unknown) => void) | null;
}

export interface Transport {
  request<T>(method: Method, path: string, body?: unknown, init?: { form?: FormData; signal?: AbortSignal }): Promise<T>;
  blob(path: string, method?: Method, body?: unknown): Promise<Blob>;
  socket(path: string): Socket;
  mediaUrl(path: string | null): string | null;
}

/* The transport mode is decided once per tab and frozen. `?mock=1` is remembered in sessionStorage so that
   client-side navigation (which drops the flag from the URL) and reloads keep the same transport; `?mock=0` clears it. */
const MOCK_MODE: boolean = (() => {
  if (typeof window === "undefined") return false;
  const fromEnv = import.meta.env.VITE_API_MOCK === "1";
  const param = new URLSearchParams(window.location.search).get("mock");
  let stored: string | null = null;
  try { stored = sessionStorage.getItem("vl-mock"); } catch { /* private mode */ }
  try { if (param === "1") sessionStorage.setItem("vl-mock", "1"); if (param === "0") sessionStorage.removeItem("vl-mock"); } catch { /* ignore */ }
  return fromEnv || param === "1" || (param !== "0" && stored === "1");
})();
export const isMock = (): boolean => MOCK_MODE;

const realTransport: Transport = {
  async request<T>(method: Method, path: string, body?: unknown, init?: { form?: FormData; signal?: AbortSignal }): Promise<T> {
    const res = await fetch(path, {
      method,
      headers: init?.form ? undefined : body !== undefined ? { "content-type": "application/json" } : undefined,
      body: init?.form ?? (body !== undefined ? JSON.stringify(body) : undefined),
      signal: init?.signal,
    });
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    let json: unknown = null;
    try { json = text ? JSON.parse(text) : null; } catch { json = null; }
    if (!res.ok) {
      const err = (json as ApiErrorBody | null)?.error ?? null;
      throw new ApiError(res.status, err, `${method} ${path} failed with ${res.status}`);
    }
    return json as T;
  },
  async blob(path, method = "GET", body) {
    const res = await fetch(path, { method, headers: body ? { "content-type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined });
    if (!res.ok) {
      let err: ApiErrorBody["error"] | null = null;
      try { err = ((await res.json()) as ApiErrorBody).error; } catch { /* not json */ }
      throw new ApiError(res.status, err, `${method} ${path} failed`);
    }
    return res.blob();
  },
  socket(path) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}${path}`);
    const s: Socket = {
      send: (d) => ws.send(d),
      close: () => ws.close(),
      onmessage: null, onclose: null, onerror: null,
    };
    ws.onmessage = (ev) => { try { s.onmessage?.(JSON.parse(ev.data as string)); } catch { /* ignore non-json */ } };
    ws.onclose = () => s.onclose?.();
    ws.onerror = (e) => s.onerror?.(e);
    return s;
  },
  mediaUrl: (p) => p,
};

let transport: Transport = realTransport;
let mockReady: Promise<void> | null = null;

/* Lazy so production bundles only pull the fixtures when mock mode is on. */
export function ensureTransport(): Promise<void> {
  if (!isMock()) return Promise.resolve();
  if (!mockReady) {
    mockReady = import("./mock/mockTransport").then((m) => { transport = m.createMockTransport(); });
  }
  return mockReady;
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => transport.request<T>("GET", path, undefined, { signal }),
  post: <T>(path: string, body?: unknown) => transport.request<T>("POST", path, body),
  put: <T>(path: string, body?: unknown) => transport.request<T>("PUT", path, body),
  del: (path: string) => transport.request<void>("DELETE", path),
  form: <T>(path: string, form: FormData) => transport.request<T>("POST", path, undefined, { form }),
  blob: (path: string, method?: Method, body?: unknown) => transport.blob(path, method, body),
  socket: (path: string) => transport.socket(path),
  mediaUrl: (path: string | null) => transport.mediaUrl(path),
};

export function qs(params: Record<string, string | number | boolean | string[] | undefined | null>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    if (Array.isArray(v)) v.forEach((x) => u.append(k, x));
    else u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : "";
}
