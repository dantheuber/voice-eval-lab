import { useEffect, useRef } from "react";
import type { Socket } from "@/api/transport";

/* Opens a socket for the lifetime of the component and routes messages to the latest handler.
   Reconnects with backoff while `enabled` is true and the server closed us unexpectedly. */
export function useSocket<M>(open: (() => Socket) | null, onMessage: (m: M, s: Socket) => void, opts: { enabled?: boolean; reconnect?: boolean; onClose?: () => void } = {}) {
  const handler = useRef(onMessage);
  handler.current = onMessage;
  const onClose = useRef(opts.onClose);
  onClose.current = opts.onClose;
  const enabled = opts.enabled ?? true;
  const reconnect = opts.reconnect ?? false;
  const sockRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!open || !enabled) return;
    let alive = true;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      const s = open();
      sockRef.current = s;
      s.onmessage = (d) => { attempt = 0; handler.current(d as M, s); };
      s.onclose = () => {
        sockRef.current = null;
        onClose.current?.();
        if (alive && reconnect) { attempt++; timer = setTimeout(connect, Math.min(10_000, 500 * 2 ** attempt)); }
      };
      s.onerror = () => { /* close follows */ };
    };
    connect();
    return () => { alive = false; clearTimeout(timer); sockRef.current?.close(); sockRef.current = null; };
  }, [enabled, reconnect, open]);

  return sockRef;
}
