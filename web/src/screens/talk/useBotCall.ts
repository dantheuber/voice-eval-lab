/* The live call to the support bot. Real mode drives Pipecat's client with the small-WebRTC transport against
   the session's offer URL. Fixture mode runs a scripted bot so the screen is explorable with no backend. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { TalkSession, VoicelabServerMessage, ToolCallEvent, BargeInEvent, SystemEvent } from "@/api/types";
import { isMock } from "@/api/transport";
import { MIC_CONSTRAINTS } from "./useMic";

export type CallStatus = "idle" | "connecting" | "live" | "ended" | "error";
export interface Line { id: string; who: "you" | "bot"; text: string; final: boolean; note?: string }
export interface CallEvent { id: string; atS: number; text: string; kind: "tool" | "barge" | "system" | "db" | "warning" }
export interface CallState {
  status: CallStatus; error: string | null; youSpeaking: boolean; botSpeaking: boolean; muted: boolean;
  lines: Line[]; events: CallEvent[]; beatsReached: string[]; replyP50S: number | null; startedAt: number | null; interruptions: number;
}
const initial = (): CallState => ({ status: "idle", error: null, youSpeaking: false, botSpeaking: false, muted: false, lines: [], events: [], beatsReached: [], replyP50S: null, startedAt: null, interruptions: 0 });

let seq = 0;
const nid = () => `l${++seq}`;

export function useBotCall(botAudioRef: React.RefObject<HTMLAudioElement | null>, micLevelRef: React.RefObject<number>) {
  const [st, setSt] = useState<CallState>(initial);
  const up = (p: Partial<CallState> | ((s: CallState) => Partial<CallState>)) => setSt((s) => ({ ...s, ...(typeof p === "function" ? p(s) : p) }));
  const clientRef = useRef<{ disconnect: () => void; mute: (on: boolean) => void } | null>(null);

  const addLine = (who: Line["who"], text: string, final: boolean) => up((s) => {
    const i = s.lines.findIndex((l) => l.who === who && !l.final);
    const line: Line = { id: i >= 0 ? s.lines[i].id : nid(), who, text, final };
    return { lines: i >= 0 ? s.lines.map((l, j) => (j === i ? line : l)) : [...s.lines, line] };
  });
  const addEvent = (kind: CallEvent["kind"], text: string) => up((s) => ({ events: [...s.events, { id: nid(), atS: s.startedAt ? (Date.now() - s.startedAt) / 1000 : 0, text, kind }] }));
  const onServer = (m: VoicelabServerMessage) => {
    if (m.type === "event") {
      const e = m.event as ToolCallEvent | BargeInEvent | SystemEvent;
      if (e.type === "tool_call") addEvent("tool", `${e.name}(${Object.entries(e.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})${e.result != null ? ` -> ${typeof e.result === "string" ? e.result : JSON.stringify(e.result)}` : ""}`);
      else if (e.type === "barge_in") { addEvent("barge", `you interrupted the bot${e.botStoppedAtS != null ? `, it stopped in ${(e.botStoppedAtS - e.atS).toFixed(1)} s` : ""}`); up((s) => ({ interruptions: s.interruptions + 1 })); }
      else addEvent(e.kind === "db_change" ? "db" : "system", e.detail || e.kind);
    } else if (m.type === "metrics") up({ replyP50S: m.replyP50S });
    else if (m.type === "beat") up((s) => ({ beatsReached: m.reached && !s.beatsReached.includes(m.beatId) ? [...s.beatsReached, m.beatId] : s.beatsReached }));
    else if (m.type === "warning") addEvent("warning", m.message);
  };

  // Each start gets a generation number; anything that resolves after end() or a newer start is torn down.
  const gen = useRef(0);
  useEffect(() => () => { gen.current++; clientRef.current?.disconnect(); clientRef.current = null; }, []);

  const start = useCallback(async (session: TalkSession, opts: { deviceId?: string | null } = {}) => {
    const my = ++gen.current;
    const stale = () => gen.current !== my;
    clientRef.current?.disconnect(); clientRef.current = null;
    setSt({ ...initial(), status: "connecting" });
    try {
      if (isMock()) { clientRef.current = startMock({ up, addLine, addEvent, onServer, micLevelRef, session }); return; }
      const [{ PipecatClient, RTVIEvent }, { SmallWebRTCTransport }] = await Promise.all([import("@pipecat-ai/client-js"), import("@pipecat-ai/small-webrtc-transport")]);
      if (stale()) return;
      const transport = new SmallWebRTCTransport({ iceServers: session.iceServers });
      let clientHandle: InstanceType<typeof PipecatClient> | null = null;
      // Captured before connect() so end() or unmount during connection still releases the mic.
      clientRef.current = {
        disconnect: () => { try { clientHandle?.disconnect(); } catch { /* not connected yet */ } },
        mute: (on) => { try { clientHandle?.enableMic(!on); clientHandle?.sendClientMessage("voicelab", { type: "mute", on }); } catch { /* not connected */ } },
      };
      const client = new PipecatClient({
        transport, enableMic: true, enableCam: false,
        callbacks: {
          onBotReady: () => up({ status: "live", startedAt: Date.now() }),
          onUserStartedSpeaking: () => up({ youSpeaking: true }),
          onUserStoppedSpeaking: () => up({ youSpeaking: false }),
          onBotStartedSpeaking: () => up({ botSpeaking: true }),
          onBotStoppedSpeaking: () => up({ botSpeaking: false }),
          onBotInterrupted: () => up({ botSpeaking: false }),
          onUserTranscript: (d) => addLine("you", d.text, d.final),
          onBotTranscript: (d) => addLine("bot", d.text, false),
          onBotTtsStopped: () => up((s) => ({ lines: s.lines.map((l) => (l.who === "bot" ? { ...l, final: true } : l)) })),
          onServerMessage: (data: unknown) => { const v = (data as { voicelab?: VoicelabServerMessage })?.voicelab; if (v) onServer(v); },
          onTrackStarted: (track, participant) => { if (!participant?.local && track.kind === "audio" && botAudioRef.current) { botAudioRef.current.srcObject = new MediaStream([track]); botAudioRef.current.play().catch(() => undefined); } },
          onDisconnected: () => up((s) => (s.status === "ended" || s.status === "error" ? {} : { status: "ended" })),
          onError: (msg) => up({ status: "error", error: String((msg as { data?: { message?: string } }).data?.message ?? "transport error") }),
        },
      });
      clientHandle = client;
      client.on(RTVIEvent.TransportStateChanged, (state) => { if (state === "error") up({ status: "error", error: "WebRTC connection failed" }); });
      // Open devices first so the selected microphone is the one Pipecat records, not the system default.
      await client.initDevices();
      if (stale()) { client.disconnect(); return; }
      if (opts.deviceId) await transport.updateMic(opts.deviceId).catch(() => undefined);
      await client.connect({ webrtcRequestParams: { endpoint: session.offerUrl }, iceConfig: { iceServers: session.iceServers } });
      if (stale()) { client.disconnect(); return; }
      // Apply the brief's constraints to Pipecat's track.
      const local = client.tracks().local.audio;
      await local?.applyConstraints(MIC_CONSTRAINTS).catch(() => undefined);
    } catch (e) {
      if (stale()) return;
      up({ status: "error", error: (e as Error).message || String(e) });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const end = useCallback(() => { gen.current++; clientRef.current?.disconnect(); clientRef.current = null; up({ status: "ended", youSpeaking: false, botSpeaking: false }); }, []);
  const setMuted = useCallback((on: boolean) => { clientRef.current?.mute(on); up({ muted: on }); }, []);
  return { state: st, start, end, setMuted };
}

/* ---------- fixture mode: scripted bot, mic level drives "you speaking" and barge-ins ---------- */
function startMock(ctx: {
  up: (p: Partial<CallState> | ((s: CallState) => Partial<CallState>)) => void; addLine: (who: Line["who"], text: string, final: boolean) => void;
  addEvent: (kind: CallEvent["kind"], text: string) => void; onServer: (m: VoicelabServerMessage) => void; micLevelRef: React.RefObject<number>; session: TalkSession;
}) {
  const { up, addLine, addEvent, onServer, micLevelRef, session } = ctx;
  const sheet = session.sheet;
  const name = sheet?.youAre.split(",")[0] ?? "there";
  const script: { who: "bot" | "you"; text: string; tool?: string; beat?: string }[] = [
    { who: "bot", text: "Thanks for calling Fernhill Internet, this is Ada. How can I help?" },
    { who: "you", text: sheet?.problem ?? "Hi, I have a question about my account." , beat: sheet?.beats[0]?.id },
    { who: "bot", text: "Sorry to hear that. Can I get the billing zip and PIN on the account?", tool: `lookup_customer(name="${name}") -> found` },
    { who: "you", text: "Sure, it's in my notes here.", beat: sheet?.beats[1]?.id },
    { who: "bot", text: `Thanks, ${name}. Let me check that for you now.`, tool: "verify_customer(...) -> verified" },
    { who: "bot", text: "I can see what's going on. I'm going to fix it on my side right now.", tool: "get_account_status(...) -> ok" },
    { who: "bot", text: "Done. Is there anything else I can help with today?" },
  ];
  let alive = true;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const later = (ms: number, fn: () => void) => { const t = setTimeout(() => { if (alive) fn(); }, ms); timers.push(t); };
  let botSpeaking = false;
  let lastYou = false;
  const replies: number[] = [];
  later(700, () => up({ status: "live", startedAt: Date.now() }));
  // mic level -> you speaking; speaking over the bot -> barge-in
  const poll = setInterval(() => {
    if (!alive) return;
    const lvl = micLevelRef.current ?? 0;
    const speaking = lvl > 0.12;
    if (speaking !== lastYou) { lastYou = speaking; up({ youSpeaking: speaking }); if (speaking && botSpeaking) { botSpeaking = false; up({ botSpeaking: false }); onServer({ type: "event", event: { type: "barge_in", id: nid(), atS: 0, botStoppedAtS: 0.3, interruptedUtteranceId: null } }); } }
  }, 100);
  let t = 1200;
  script.forEach((turn, i) => {
    const dur = 900 + turn.text.length * 45;
    later(t, () => {
      if (turn.who === "bot") {
        if (turn.tool) onServer({ type: "event", event: { type: "tool_call", id: nid(), atS: 0, name: turn.tool.split("(")[0], arguments: { note: turn.tool.slice(turn.tool.indexOf("(") + 1, turn.tool.indexOf(")")) }, result: turn.tool.split("->")[1]?.trim() ?? "ok", error: null, durationMs: 80 } });
        botSpeaking = true; up({ botSpeaking: true });
        let n = 0; const step = () => { if (!alive || !botSpeaking) { addLine("bot", turn.text, true); return; } n = Math.min(turn.text.length, n + 3); addLine("bot", turn.text.slice(0, n), n >= turn.text.length); if (n < turn.text.length) later(45, step); else { botSpeaking = false; up({ botSpeaking: false }); replies.push(1.0 + Math.random() * 0.6); onServer({ type: "metrics", replyP50S: +(replies.reduce((a, b) => a + b, 0) / replies.length).toFixed(2), lastReplyS: replies[replies.length - 1] }); } };
        step();
      } else {
        addLine("you", turn.text, true);
        if (turn.beat) onServer({ type: "beat", beatId: turn.beat, reached: true });
        if (i === 1) addEvent("warning", "Fixture mode: your speech is not transcribed. The level meter and barge-in detection use your real microphone.");
      }
    });
    t += dur + 900;
  });
  return {
    disconnect: () => { alive = false; timers.forEach(clearTimeout); clearInterval(poll); },
    mute: () => undefined,
  };
}
