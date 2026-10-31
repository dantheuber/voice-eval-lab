import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalMic, asMediaManager } from "./localMic";

// The transport bundle builds audio worklets and probes the canvas as it loads, and jsdom has neither.
vi.hoisted(() => {
  URL.createObjectURL = () => "blob:stub";
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
});

interface FakeTrack { kind: string; enabled: boolean; label: string; stop: ReturnType<typeof vi.fn>; getSettings: () => { deviceId: string } }

/* getUserMedia that answers when the test says so, as a browser does while its permission prompt is up. */
function fakeMedia({ answerAtOnce = false } = {}) {
  const waiting: ((stream: MediaStream) => void)[] = [];
  const tracks: FakeTrack[] = [];
  const stream = () => {
    const track: FakeTrack = { kind: "audio", enabled: true, label: "Fake mic", stop: vi.fn(), getSettings: () => ({ deviceId: "mic-1" }) };
    tracks.push(track);
    return { getAudioTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream;
  };
  const getUserMedia = vi.fn(() => (answerAtOnce ? Promise.resolve(stream()) : new Promise<MediaStream>((resolve) => waiting.push(resolve))));
  const enumerateDevices = vi.fn(async () => [{ deviceId: "mic-1", groupId: "g", kind: "audioinput", label: "Fake mic" }] as MediaDeviceInfo[]);
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia, enumerateDevices, addEventListener: vi.fn(), removeEventListener: vi.fn() } });
  return { getUserMedia, enumerateDevices, tracks, answer: () => { waiting.shift()!(stream()); return tracks[tracks.length - 1]; } };
}

const live = (media: ReturnType<typeof fakeMedia>) => media.tracks.filter((t) => t.stop.mock.calls.length === 0);
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

function pipecat(mic: LocalMic, callbacks = {}) {
  const transport = new SmallWebRTCTransport({ mediaManager: asMediaManager(mic) });
  return new PipecatClient({ transport, enableMic: true, enableCam: false, callbacks });
}

const options = (callbacks = {}) => ({ enableMic: true, callbacks }) as Parameters<LocalMic["setClientOptions"]>[0];

describe("LocalMic with the real Pipecat client", () => {
  afterEach(() => vi.restoreAllMocks());
  beforeEach(() => { vi.spyOn(console, "warn").mockImplementation(() => undefined); });

  it("opens the mic once, because the client hears that it's open", async () => {
    const media = fakeMedia({ answerAtOnce: true });
    const client = pipecat(new LocalMic());
    await client.initDevices();
    await client.connect({ webrtcRequestParams: { endpoint: "/api/talk/sessions/x/offer" } }).catch(() => undefined);
    expect(media.getUserMedia).toHaveBeenCalledTimes(1);
  });

  it("has no mic left open when End lands while connect is setting up", async () => {
    const media = fakeMedia();
    const mic = new LocalMic();
    const client = pipecat(mic);
    const init = client.initDevices();
    await settle();
    media.answer();
    await init;
    const connecting = client.connect({ webrtcRequestParams: { endpoint: "/api/talk/sessions/x/offer" } }).catch(() => undefined);
    await settle();
    void client.disconnect();
    mic.stop();
    while (media.getUserMedia.mock.calls.length > media.tracks.length) media.answer();
    await settle();
    await connecting;
    expect(live(media)).toEqual([]);
  });
});

describe("LocalMic on its own", () => {
  it("stops a stream that arrives after stop", async () => {
    const media = fakeMedia();
    const mic = new LocalMic();
    const opening = mic.initialize();
    mic.stop();
    const late = media.answer();
    await opening;
    expect(late.stop).toHaveBeenCalled();
    expect(mic.tracks().local).toEqual({});
  });

  it("stays stopped while the device list is coming back", async () => {
    const media = fakeMedia();
    const mic = new LocalMic();
    const opened = vi.fn();
    mic.setClientOptions(options({ onMicUpdated: opened }));
    const opening = mic.initialize();
    media.answer();
    await Promise.resolve();
    mic.stop();
    await opening;
    expect(live(media)).toEqual([]);
    expect(opened).not.toHaveBeenCalled();
  });

  it("never opens again once stopped", async () => {
    const media = fakeMedia({ answerAtOnce: true });
    const mic = new LocalMic();
    mic.stop();
    await mic.connect();
    await mic.initialize();
    expect(media.getUserMedia).not.toHaveBeenCalled();
  });

  it("keeps only the newest of two opens", async () => {
    const media = fakeMedia();
    const mic = new LocalMic();
    const first = mic.initialize();
    const second = mic.updateMic("mic-2");
    const older = media.answer();
    const newer = media.answer();
    await Promise.all([first, second]);
    expect(older.stop).toHaveBeenCalled();
    expect(newer.stop).not.toHaveBeenCalled();
    expect(mic.tracks().local).toEqual({ audio: newer });
  });

  it("reports the device, the list, and the track, as the transport's own managers do", async () => {
    const media = fakeMedia({ answerAtOnce: true });
    const seen = { mic: vi.fn(), mics: vi.fn(), started: vi.fn(), stopped: vi.fn() };
    const mic = new LocalMic();
    mic.setClientOptions(options({ onMicUpdated: seen.mic, onAvailableMicsUpdated: seen.mics, onTrackStarted: seen.started, onTrackStopped: seen.stopped }));
    await mic.initialize();
    expect(seen.mic).toHaveBeenCalledWith(expect.objectContaining({ deviceId: "mic-1" }));
    expect(seen.mics).toHaveBeenCalledWith([expect.objectContaining({ deviceId: "mic-1" })]);
    const local = expect.objectContaining({ local: true });
    expect(seen.started).toHaveBeenCalledWith(media.tracks[0], local);
    mic.stop();
    expect(seen.stopped).toHaveBeenCalledWith(media.tracks[0], local);
  });

  it("reports a device id even when the browser doesn't list the track's device", async () => {
    const media = fakeMedia({ answerAtOnce: true });
    media.enumerateDevices.mockResolvedValue([]);
    const opened = vi.fn();
    const mic = new LocalMic();
    mic.setClientOptions(options({ onMicUpdated: opened }));
    await mic.initialize();
    expect(opened).toHaveBeenCalledWith(expect.objectContaining({ deviceId: "mic-1" }));
  });
});
