/* Types matching docs/API.md. Keep the two in step. */

export type CallMode = "live_bots" | "turn_based" | "human" | "upload";
export type CallStatus = "queued" | "running" | "evaluating" | "done" | "failed";
export type Verdict = "pass" | "fail" | "needs_review" | "invalid";
export type Speaker = "caller" | "bot";
export type Severity = "info" | "minor" | "major" | "critical";
export type HumanVerdict = "confirmed" | "dismissed";
export type HealthState = "ok" | "loading" | "down" | "off";

export interface LineConditions {
  codec: "none" | "mulaw_8k";
  noise: "none" | "office" | "street" | "cafe" | "car" | "tv" | "hum";
  snrDb: number;
  packetLossPct: number;
  oneWayLatencyMs: number;
}
export interface VoiceRef { provider: string | null; voiceId: string; style: string | null }

export interface Persona {
  id: string; name: string; summary: string; age: number | null; temperament: string;
  speakingStyle: string; patience: number; verbosity: number; techSavviness: number;
  interruptionRate: number; voice: VoiceRef; line: LineConditions; createdAt: string; updatedAt: string;
}
export type PersonaInput = Omit<Persona, "id" | "createdAt" | "updatedAt">;

export interface Beat { id: string; instruction: string; checkpoint: string | null; required: boolean }
export interface Rule { id: string; text: string; severity: "critical" | "major" | "minor"; needsBeat?: string | null; decidedBy?: string | null }
export interface StateExpectation {
  id: string; table: string; where: Record<string, unknown>; op: "changed" | "no_change" | "row_added";
  changes: Record<string, unknown>; description: string; needsBeat?: string | null;
}
export interface Scenario {
  id: string; title: string; description: string; goal: string; customerId: string | null;
  facts: Record<string, string>; beats: Beat[]; must: Rule[]; mustNot: Rule[]; expectedState: StateExpectation[];
  maxTurns: number; pronunciationWatch: string[]; tags: string[]; createdAt: string; updatedAt: string;
}
export type ScenarioInput = Omit<Scenario, "id" | "createdAt" | "updatedAt">;

export interface Word { text: string; startS: number; endS: number; confidence: number | null }
export interface Utterance {
  type: "utterance"; id: string; speaker: Speaker; startS: number; endS: number;
  intendedText: string | null; spokenText: string | null; heardText: string | null; interrupted: boolean; ttfbMs: number | null; words?: Word[];
}
export interface ToolCallEvent {
  type: "tool_call"; id: string; atS: number; name: string; arguments: Record<string, unknown>; result: unknown; error: string | null; durationMs: number | null;
}
export interface BargeInEvent { type: "barge_in"; id: string; atS: number; botStoppedAtS: number | null; interruptedUtteranceId: string | null }
export interface SystemEvent { type: "system"; id: string; atS: number; kind: string; detail: string }
export type TimelineEvent = Utterance | ToolCallEvent | BargeInEvent | SystemEvent;
export interface DbChange {
  table: string; key: Record<string, unknown>; kind: "insert" | "update" | "delete";
  before: Record<string, unknown> | null; after: Record<string, unknown> | null;
}

export interface CallArtifact {
  id: string; mode: CallMode; status: CallStatus; timingIsReal: boolean; batchId: string | null;
  personaId: string | null; scenarioId: string | null; personaSnapshot: Persona | null; scenarioSnapshot: Scenario | null;
  profile: string | null; providers: Record<string, string>; startedAt: string | null; endedAt: string | null; durationS: number;
  timeline: TimelineEvent[]; dbChanges: DbChange[]; hasToolLog: boolean; hasIntendedText: boolean; error: string | null;
}

export interface Evidence { quote: string; speaker: Speaker | null; startS: number | null; endS: number | null }
export interface Finding {
  id: string; evaluator: string; category: string; severity: Severity; advisory: boolean; title: string; detail: string;
  evidence: Evidence[]; clipStartS: number | null; clipEndS: number | null; criterionId: string | null;
  humanVerdict: HumanVerdict | null; humanNote: string; humanAt: string | null;
}
export interface CriterionResult {
  criterionId: string; kind: "must" | "must_not" | "beat" | "state" | "general"; text: string;
  status: "pass" | "fail" | "unknown"; reason: string; evidence: Evidence[];
}
export interface EvaluatorRun { name: string; status: "ok" | "skipped" | "error"; reason: string; durationS: number; model: string | null; metrics: Record<string, unknown> }
export interface EvalResult { callId: string; verdict: Verdict; summary: string; criteria: CriterionResult[]; findings: Finding[]; evaluators: EvaluatorRun[]; completedAt: string }

export interface Counts { pass: number; fail: number; needsReview: number; invalid: number }

export interface CallSummary {
  id: string; batchId: string | null; rep: number | null; scenarioId: string | null; scenarioTitle: string | null;
  personaId: string | null; personaName: string | null; mode: CallMode; status: CallStatus; profile: string | null;
  verdict: Verdict | null; humanVerdict: Verdict | null; invalidReason: string | null;
  findings: number; openFindings: number; durationS: number; replyP50S: number | null; overlaps: number | null;
  startedAt: string | null; timingIsReal: boolean;
  untested?: boolean; // true when any adapter used was never tested against a live provider (Bedrock)
}
export type RunStatus = "queued" | "running" | "paused" | "stopped" | "done";
export interface RunSummary {
  id: string; label: string; profile: string; mode: CallMode; status: RunStatus;
  startedAt: string; endedAt: string | null; calls: number; done: number; running: number; counts: Counts; etaS: number | null;
  scenarioIds: string[]; personaIds: string[]; reps: number;
  untested?: boolean;
}
export interface Health { role: string; name: string; model: string; state: HealthState; detail: string; elapsedS: number | null; usualS: number | null }
export interface Profile {
  id: string; label: string; description: string; configPath: string; health: HealthState; liveCapable: boolean; maxCalls?: number | null;
  benchmark: { botReplyP50S: number; at: string } | null; untested: boolean;
}

// Overview
export interface QueueItem { callId: string; scenarioId: string | null; scenarioTitle: string | null; personaName: string | null; openFindings: number; why: string; createdAt: string }
export interface Overview {
  window: string; profile: string; counts: Counts; queue: QueueItem[]; queueTotal: number; openFindingsTotal: number;
  reviewPaceSPerFinding: number | null; callsPerHour: number[]; recentRuns: RunSummary[]; runningRuns: number;
  scenarioRates: { scenarioId: string; scenarioTitle: string; counts: Counts }[]; health: Health[];
  topOpenFindings: { title: string; evaluator: string; count: number }[];
}

// Calls
export interface CallsQuery {
  verdict?: Verdict[]; scenario?: string; persona?: string; mode?: CallMode; profile?: string; run?: string;
  openOnly?: boolean; status?: CallStatus; cursor?: string; limit?: number;
}
export interface CallsPage { items: CallSummary[]; nextCursor: string | null; total: number; facets: { verdict: Record<string, number> } }

export interface EvidenceFlags { audio: boolean; transcript: boolean; intendedText: boolean; toolLog: boolean; dbSnapshots: boolean; scenario: boolean }
export interface Timing { replyP50S: number; replyP95S: number; overlaps: number; deadAirEvents: number; bargeStopS: number | null }
export interface ReviewPayload {
  call: CallArtifact; eval: EvalResult | null; summary: CallSummary;
  audio: { stereoUrl: string | null; callerUrl: string | null; botUrl: string | null; sampleRate: number; available: boolean };
  peaks: { perSecond: number; caller: number[]; bot: number[] } | null;
  speech: { caller: [number, number][]; bot: [number, number][] };
  overlaps: [number, number][];
  timing: Timing | null; evidence: EvidenceFlags; prevReviewId: string | null; nextReviewId: string | null;
}
export type CallWsMessage =
  | { type: "status"; status: CallStatus }
  | { type: "evaluator"; evaluator: EvaluatorRun }
  | { type: "finding"; finding: Finding }
  | { type: "eval"; eval: EvalResult }
  | { type: "error"; message: string };

// Personas and scenarios
export interface Usage { calls: number; counts: Counts }
export type PersonaListItem = Persona & { usage: Usage };
export type PersonaDetail = Persona & { usage: Usage; recentCalls: CallSummary[] };
export interface Voice { provider: string; voiceId: string; label: string; language: string }
export type ScenarioListItem = Scenario & { usage: Usage; version: number };
export type ScenarioDetail = Scenario & { usage: Usage; version: number; recentCalls: CallSummary[] };
export interface Customer { id: string; name: string; note: string; plans: string[]; domain: string | null }
export interface ScenarioDraft { scenario: ScenarioInput; guessedFields: string[]; notes: string; model: string }
export interface DryRun { turns: { speaker: Speaker; text: string }[]; reachedCheckpoints: string[]; model: string }

// Runs
export interface MatrixCell { scenarioId: string; personaId: string; counts: Counts }
export interface RunRequest {
  pairs: { scenarioId: string; personaId: string }[]; reps: number; profile: string;
  mode: "auto" | "live_bots" | "turn_based"; line: "persona" | LineConditions; concurrency: number; label: string | null;
}
export interface RunCreated { run: RunSummary; effectiveMode: CallMode; estimateS: number }
export interface RunDetail { run: RunSummary; calls: CallSummary[] }
export type RunWsMessage =
  | { type: "run"; run: RunSummary }
  | { type: "callStarted"; call: CallSummary }
  | { type: "partial"; callId: string; speaker: Speaker; text: string; atS: number }
  | { type: "turn"; callId: string; utterance: Utterance }
  | { type: "event"; callId: string; event: ToolCallEvent | BargeInEvent | SystemEvent }
  | { type: "callDone"; call: CallSummary }
  | { type: "callEvaluated"; call: CallSummary }
  | { type: "runDone"; run: RunSummary };

// Talk
export interface TalkSheet { youAre: string; problem: string; beats: Beat[]; mustNotSay: string[] }
export interface TalkSession {
  sessionId: string; callId: string; offerUrl: string; iceServers: RTCIceServer[]; sheet: TalkSheet | null;
}
export interface TalkSessionState { sessionId: string; callId: string; state: "connecting" | "live" | "ended"; startedAt: string; durationS: number }
export type VoicelabServerMessage =
  | { type: "event"; event: ToolCallEvent | BargeInEvent | SystemEvent }
  | { type: "metrics"; replyP50S: number; lastReplyS: number }
  | { type: "warning"; code: "echo_suspected" | "no_headphones"; message: string };

// Upload
export interface UploadProbe { durationS: number; channels: 1 | 2; sampleRate: number; channelMap: "caller_left" | "caller_right" | "mono" | "unknown"; sizeBytes: number }
export interface UploadResult { callId: string; probe: UploadProbe; evidence: EvidenceFlags }

// Settings
export interface RoleOption { id: string; label: string; untested: boolean; license: string }
export interface RoleItem { role: string; label: string; interface: string; adapter: string; options: RoleOption[]; health: Health }
export interface SystemInfo {
  gpu: { device: string; memoryUsedGb: number; memoryTotalGb: number };
  servers: { role: string; kind: string; port: number; pid: number | null }[];
  urls: { local: string; lan: string }; cert: string;
}
export interface Thresholds { deadAirS: number; bargeStopTargetS: number; replyLatencyWarnS: number; mosBaseline: number; audioListeningMode: "advisory" }
export interface AwsStatus { region: string | null; credentials: "found" | "not_found"; testedAgainstLive: boolean }

// Compare
export interface Frac { pass: number; n: number }
export interface Compare {
  a: RunSummary; b: RunSummary; scenarioIds: string[]; droppedScenarioIds: string[];
  criteria: { criterionId: string; text: string; kind: CriterionResult["kind"]; a: Frac; b: Frac }[];
  measures: { key: string; label: string; a: number | null; b: number | null; comparable: boolean; reason?: string; lowerIsBetter?: boolean }[];
  cells: { scenarioId: string; personaId: string; a: Frac; b: Frac }[];
  worst: { criterionId: string; deltaPts: number; callIds: string[] } | null;
}

export interface ApiErrorBody { error: { code: string; message: string; detail?: Record<string, unknown> } }
