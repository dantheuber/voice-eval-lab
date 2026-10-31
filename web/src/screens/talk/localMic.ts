/* The microphone for a live call. The transport's default media manager is Daily's, which downloads its call machine from
   Daily's CDN when a call starts, so a call to a bot on this machine needed the internet. All the transport needs is one
   audio track. This also opens the track with our constraints, rather than opening it with defaults and adjusting it. */
import type { PipecatClientOptions } from "@pipecat-ai/client-js";
import type { SmallWebRTCTransportConstructorOptions } from "@pipecat-ai/small-webrtc-transport";
import { MIC_CONSTRAINTS } from "./useMic";

type MediaManager = NonNullable<SmallWebRTCTransportConstructorOptions["mediaManager"]>;

export class LocalMic {
  private track: MediaStreamTrack | null = null;
  private device: MediaDeviceInfo | null = null;
  private enabled = true;

  constructor(private deviceId: string | null = null) {}

  private async open() {
    this.stop();
    const audio = this.deviceId ? { ...MIC_CONSTRAINTS, deviceId: { exact: this.deviceId } } : MIC_CONSTRAINTS;
    const stream = await navigator.mediaDevices.getUserMedia({ audio });
    this.track = stream.getAudioTracks()[0];
    this.track.enabled = this.enabled;
    const id = this.track.getSettings().deviceId;
    this.device = (await this.mics()).find((d) => d.deviceId === id) ?? null;
  }

  private async mics() {
    return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  }

  /** Releases the microphone. Safe to call at any point, and the browser's mic indicator goes out with it. */
  stop() {
    this.track?.stop();
    this.track = null;
  }

  // What the transport calls.
  setClientOptions(options: PipecatClientOptions) { this.enabled = options.enableMic ?? true; }
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
