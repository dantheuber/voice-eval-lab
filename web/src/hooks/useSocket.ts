import { useEffect, useRef } from "react";
import type { Socket } from "@/api/transport";

/* Opens a socket for the lifetime of the component and routes messages to the latest handler.
   Reconnects with backoff while `enabled` is true and the server closed us unexpectedly. Callbacks are
   guarded by socket identity, so a stale socket's late close can never clear or close its replacement. */
export function useSocket<M>(
  open: (() => Socket) | null,
  onMessage: (m: M, s: Socket) => void,
  opts: { enabled?: boolean; reconnect?: boolean; onClose?: () => void; onOpen?: (s: Socket, attempt: number) => void } = {},
) {
  const handler = useRef(onMessage);
  handler.current = onMessage;
  const onClose = useRef(opts.onClose);
  onClose.current = opts.onClose;
  const onOpen = useRef(opts.onOpen);
  onOpen.current = opts.onOpen;
  const enabled = opts.enabled ?? true;
  const reconnect = opts.reconnect ?? false;
  const sockRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!open || !enabled) return;
    let alive = true;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let current: Socket | null = null;
    const connect = () => {
      const s = open();
      current = s;
      sockRef.current = s;
      onOpen.current?.(s, attempt);
      s.onmessage = (d) => { if (!alive || current !== s) return; attempt = 0; handler.current(d as M, s); };
      s.onclose = () => {
        if (current !== s) return; // a replacement already took over
        current = null;
        if (sockRef.current === s) sockRef.current = null;
        if (!alive) return;
        onClose.current?.();
        if (reconnect) { attempt++; timer = setTimeout(connect, Math.min(10_000, 500 * 2 ** attempt)); }
      };
      s.onerror = () => { /* close follows */ };
    };
    connect();
    return () => {
      alive = false;
      clearTimeout(timer);
      const s = current;
      current = null;
      if (sockRef.current === s) sockRef.current = null;
      s?.close();
    };
  }, [enabled, reconnect, open]);

  return sockRef;
}
