/* The microphone for a live call. The transport's default media manager is Daily's, which downloads its call machine from
   Daily's CDN when a call starts, so a call to a bot on this machine needed the internet. All the transport needs is one
   audio track. This also opens the track with our constraints, rather than opening it with defaults and adjusting it.

   A LocalMic lasts one call. Once stopped it never opens again, and a stream the browser hands over after that is stopped
   at once, because getUserMedia can answer long after someone pressed End. It also has to tell the client what it opened:
   a client that hears nothing assumes the mic was never set up and opens it again inside connect(). */
import type { PipecatClientOptions } from "@pipecat-ai/client-js";
import type { SmallWebRTCTransportConstructorOptions } from "@pipecat-ai/small-webrtc-transport";
import { MIC_CONSTRAINTS } from "./useMic";

type MediaManager = NonNullable<SmallWebRTCTransportConstructorOptions["mediaManager"]>;
type Callbacks = NonNullable<PipecatClientOptions["callbacks"]>;

const LOCAL = { id: "local", name: "", local: true };

export class LocalMic {
  private track: MediaStreamTrack | null = null;
  private device: MediaDeviceInfo | null = null;
  private enabled = true;
  private callbacks: Callbacks = {};
  private stopped = false;
  private opening = 0; // moves on with every open and stop, so an older open can tell it's stale

  constructor(private deviceId: string | null = null) {}

  private async open() {
    if (this.stopped) return;
    this.release();
    const mine = ++this.opening;
    const audio = this.deviceId ? { ...MIC_CONSTRAINTS, deviceId: { exact: this.deviceId } } : MIC_CONSTRAINTS;
    const stream = await navigator.mediaDevices.getUserMedia({ audio });
    if (mine !== this.opening) { stream.getTracks().forEach((t) => t.stop()); return; }
    const track = stream.getAudioTracks()[0];
    this.track = track;
    track.enabled = this.enabled;
    const id = track.getSettings().deviceId;
    const mics = await this.mics();
    if (mine !== this.opening) return; // whatever moved it on has already stopped this track
    this.device = mics.find((d) => d.deviceId === id) ?? null;
    this.callbacks.onAvailableMicsUpdated?.(mics);
    this.callbacks.onMicUpdated?.(this.device ?? ({ deviceId: id || "default", groupId: "", kind: "audioinput", label: track.label } as MediaDeviceInfo));
    this.callbacks.onTrackStarted?.(track, LOCAL);
  }

  private release() {
    const track = this.track;
    this.track = null;
    if (!track) return;
    track.stop();
    this.callbacks.onTrackStopped?.(track, LOCAL);
  }

  private async mics() {
    return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  }

  /** Releases the microphone for good. Safe to call at any point, and the browser's mic indicator goes out with it. */
  stop() {
    this.stopped = true;
    this.opening++;
    this.release();
  }

  // What the transport calls.
  setClientOptions(options: PipecatClientOptions) { this.enabled = options.enableMic ?? true; this.callbacks = options.callbacks ?? {}; }
  setUserAudioCallback() {}
  async initialize() { await this.open(); }
  async connect() { if (!this.track) await this.open(); }
  async disconnect() { this.stop(); }
  async interruptBotAudio() {}
  bufferBotAudio() { return undefined; }
  getAllMics() { return this.mics(); }
  getAllCams() { return Promise.resolve([]); }
  getAllSpeakers() { return Promise.resolve([]); }
  updateMic(id: string) { this.deviceId = id; return this.open(); }
  updateCam() {}
  updateSpeaker() {}
  enableMic(on: boolean) { this.enabled = on; if (this.track) this.track.enabled = on; }
  enableCam() {}
  enableScreenShare() {}
  tracks() { return { local: this.track ? { audio: this.track } : {} }; }
  get selectedMic() { return this.device ?? {}; }
  get selectedCam() { return {}; }
  get selectedSpeaker() { return {}; }
  get isMicEnabled() { return this.enabled; }
  get isCamEnabled() { return false; }
  get isSharingScreen() { return false; }
  get supportsScreenShare() { return false; }
  get mediaSupport() { return { mic: true, cam: false, screenShare: false }; }
}

export const asMediaManager = (mic: LocalMic) => mic as unknown as MediaManager;
