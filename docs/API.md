# Dashboard API contract

This is the contract between `web/` and the FastAPI app in `src/voicelab/api`. The frontend owns this file. The backend implements exactly these routes and shapes. Change it here first, then on both sides.

## Conventions

- Base paths: REST under `/api`, WebSockets under `/ws`, same origin as the dashboard. In development Vite proxies both to `http://127.0.0.1:18080`.
- JSON everywhere, **camelCase field names**. Every Pydantic model in `src/voicelab/core/models.py` is serialised with its snake_case fields converted mechanically to camelCase (`start_s` → `startS`, `clip_start_s` → `clipStartS`, `must_not` → `mustNot`, `created_at` → `createdAt`). Use `alias_generator=to_camel, populate_by_name=True` on the shared base model and nothing else needs renaming. Enum values stay as they are in the models (`live_bots`, `needs_review`, `transcript_judge`).
- Times inside a call are seconds from call start, floats, field names end in `S` (`startS`, `atS`, `durationS`). Wall-clock times are ISO 8601 strings with a timezone (`2026-10-30T09:17:08Z`).
- Nullable means "not known or not measured". A null timing value is rendered as "n/a" with a reason. Never send `0` to mean "not measured".
- Counts, not percentages. The UI computes pass rates and Wilson intervals from `{ pass, fail, needsReview, invalid }`. Invalid is always returned separately and never folded into fail.
- Lists that can grow are paginated with `?cursor=&limit=` and return `{ items, nextCursor }`. `nextCursor` is null on the last page.
- Errors: non-2xx with `{ "error": { "code": "not_found" | "invalid" | "conflict" | "profile_busy" | "not_installed" | "profile_failed" | "model_not_ready" | "internal", "message": "...", "detail": {...} } }`. `model_not_ready` carries `detail: { role, model, elapsedS, usualS }` so the UI can show the loading state. `profile_busy` is a 409 that carries `detail: { profile, maxCalls, waitS }`, and `waitS` is null when the run has no finished call to estimate from. `not_installed` is a 400 that carries `detail: { profile, missing, setup }`, where `setup` is the command to run, or null if setup doesn't cover the profile. `profile_failed` is a 503 from a swap whose servers didn't start.
- IDs are the model ids: `call_…`, `per_…`, `scn_…`, `fnd_…`, `batch_…`. The UI calls a batch a "run"; the API uses `run` in paths and `batchId` in fields, both meaning the same `batch_id` from `CallArtifact`.
- Mutations are `POST` for create and actions, `PUT` for full replace. Responses return the updated resource.

## Shared types

Pydantic models, camelCased. Listed here so field names are unambiguous.

```ts
type CallMode = "live_bots" | "turn_based" | "human" | "upload";
type CallStatus = "queued" | "running" | "evaluating" | "done" | "failed";
type Verdict = "pass" | "fail" | "needs_review" | "invalid";
type Speaker = "caller" | "bot";
type Severity = "info" | "minor" | "major" | "critical";
type HumanVerdict = "confirmed" | "dismissed";
type HealthState = "ok" | "loading" | "down" | "off";

interface LineConditions { codec: "none" | "mulaw_8k"; noise: "none"|"office"|"street"|"cafe"|"car"|"tv"|"hum"; snrDb: number; packetLossPct: number; oneWayLatencyMs: number; }
interface VoiceRef { provider: string | null; voiceId: string; style: string | null; }

interface Persona {
  id: string; name: string; summary: string; age: number | null; temperament: string;
  speakingStyle: string; patience: 1|2|3|4|5; verbosity: 1|2|3|4|5; techSavviness: 1|2|3|4|5;
  interruptionRate: number;            // 0..1, chance per bot turn. UI shows it as a 1..5 scale: 0, .25, .5, .75, 1
  voice: VoiceRef; line: LineConditions; createdAt: string; updatedAt: string;
}

interface Beat { id: string; instruction: string; checkpoint: string | null; required: boolean; }
interface Rule { id: string; text: string; severity: "critical" | "major" | "minor"; needsBeat?: string | null; decidedBy?: string | null; }
interface StateExpectation { id: string; table: string; where: Record<string, unknown>; op: "changed" | "no_change" | "row_added"; changes: Record<string, unknown>; description: string; }

interface Scenario {
  id: string; title: string; description: string; goal: string; customerId: string | null;
  facts: Record<string, string>; beats: Beat[]; must: Rule[]; mustNot: Rule[]; expectedState: StateExpectation[];
  maxTurns: number; pronunciationWatch: string[]; tags: string[]; createdAt: string; updatedAt: string;
}

interface Word { text: string; startS: number; endS: number; confidence: number | null; }
interface Utterance { type: "utterance"; id: string; speaker: Speaker; startS: number; endS: number; intendedText: string | null; spokenText: string | null; heardText: string | null; interrupted: boolean; ttfbMs: number | null; words?: Word[]; }
interface GuardObjection { rule: string; fields: string[]; stopped: boolean; } // rule is not_heard or digit_count. stopped is false when the guard was off and the call ran
interface ToolCallEvent { type: "tool_call"; id: string; atS: number; name: string; arguments: Record<string, unknown>; result: unknown; error: string | null; durationMs: number | null; guard: GuardObjection | null; }
interface BargeInEvent { type: "barge_in"; id: string; atS: number; botStoppedAtS: number | null; interruptedUtteranceId: string | null; }
interface SystemEvent { type: "system"; id: string; atS: number; kind: string; detail: string; }
type TimelineEvent = Utterance | ToolCallEvent | BargeInEvent | SystemEvent;
interface DbChange { table: string; key: Record<string, unknown>; kind: "insert" | "update" | "delete"; before: Record<string, unknown> | null; after: Record<string, unknown> | null; }

interface CallArtifact {
  id: string; mode: CallMode; status: CallStatus; timingIsReal: boolean; batchId: string | null;
  personaId: string | null; scenarioId: string | null; personaSnapshot: Persona | null; scenarioSnapshot: Scenario | null;
  profile: string | null; providers: Record<string, string>; startedAt: string | null; endedAt: string | null; durationS: number;
  timeline: TimelineEvent[]; dbChanges: DbChange[]; hasToolLog: boolean; hasIntendedText: boolean; error: string | null;
  metadata: Record<string, unknown>; // keys keep their spelling. `argument_guard` is the setting a call ran with, absent on realtime calls and older ones
}

interface Evidence { quote: string; speaker: Speaker | null; startS: number | null; endS: number | null; }
interface Finding {
  id: string; evaluator: string; category: string; severity: Severity; advisory: boolean; title: string; detail: string;
  evidence: Evidence[]; clipStartS: number | null; clipEndS: number | null; criterionId: string | null;
  humanVerdict: HumanVerdict | null; humanNote: string; humanAt: string | null;
}
interface CriterionResult { criterionId: string; kind: "must" | "must_not" | "beat" | "state" | "general"; text: string; status: "pass" | "fail" | "unknown"; reason: string; evidence: Evidence[]; }
interface EvaluatorRun { name: string; status: "ok" | "skipped" | "error"; reason: string; durationS: number; model: string | null; metrics: Record<string, unknown>; }
interface EvalResult { callId: string; verdict: Verdict; summary: string; criteria: CriterionResult[]; findings: Finding[]; evaluators: EvaluatorRun[]; completedAt: string; }
```

Types that exist only in the API (no Pydantic model yet):

```ts
interface Counts { pass: number; fail: number; needsReview: number; invalid: number; }   // valid = pass + fail + needsReview

interface CallSummary {                       // one row of the calls list
  id: string; batchId: string | null; rep: number | null; scenarioId: string | null; scenarioTitle: string | null;
  personaId: string | null; personaName: string | null; mode: CallMode; status: CallStatus; profile: string | null;
  verdict: Verdict | null;                    // null until evaluated
  humanVerdict: Verdict | null;               // set from the review screen's verdict bar
  invalidReason: string | null;
  findings: number; openFindings: number;     // openFindings excludes advisory findings
  durationS: number; replyP50S: number | null; overlaps: number | null;   // null when timingIsReal is false
  startedAt: string | null; timingIsReal: boolean;
  untested: boolean;                          // true when any provider in CallArtifact.providers is an adapter never tested live (bedrock:*)
}

interface RunSummary {
  id: string; label: string; profile: string; mode: CallMode; status: "queued" | "running" | "paused" | "stopped" | "done";
  startedAt: string; endedAt: string | null; calls: number; done: number; running: number; counts: Counts; etaS: number | null;
  scenarioIds: string[]; personaIds: string[]; reps: number;
  untested: boolean;                          // the profile's adapters at launch time, preserved even if the profile is edited later
}

interface Health { role: string; name: string; model: string; state: HealthState; detail: string; elapsedS: number | null; usualS: number | null; }

interface Profile { id: string; label: string; description: string; configPath: string; health: HealthState; liveCapable: boolean; maxCalls: number | null; benchmark: { botReplyP50S: number; at: string } | null; untested: boolean; }
```

## Overview

`GET /api/overview?profile=<id>&window=24h`

```json
{
  "window": "24h", "profile": "local-default",
  "counts": { "pass": 73, "fail": 21, "needsReview": 13, "invalid": 4 },
  "queue": [ { "callId": "call_a3f2…", "scenarioId": "scn_double_charge_refund", "scenarioTitle": "Double charge refund", "personaName": "Impatient small business owner", "openFindings": 4, "why": "wrong refund timeframe, no refund issued", "createdAt": "2026-10-30T09:21:08Z" } ],
  "queueTotal": 7, "openFindingsTotal": 20, "reviewPaceSPerFinding": 75,
  "callsPerHour": [0,0,0,0,0,0,0,1,4,9,12,6,2,0,0,2,8,10,5,3,1,0,0,0],
  "recentRuns": [ RunSummary ],
  "runningRuns": 0,
  "scenarioRates": [ { "scenarioId": "scn_dns_mx_fix", "scenarioTitle": "MX record fix", "counts": Counts } ],
  "health": [ Health ],
  "topOpenFindings": [ { "title": "Invoice or reference number mispronounced", "evaluator": "pronunciation", "count": 14 } ]
}
```

`queue` is the next 8 calls needing review, oldest first. `reviewPaceSPerFinding` is the median seconds between the reviewer's last 50 finding decisions, or null.

## Calls

`GET /api/calls?verdict=&scenario=&persona=&mode=&profile=&run=&openOnly=true&status=&cursor=&limit=50`
→ `{ "items": CallSummary[], "nextCursor": string | null, "total": number, "facets": { "verdict": { "pass": 3, "fail": 3, "needs_review": 5, "invalid": 1 }, "all": 14 } }`

Repeating a filter (`?verdict=pass&verdict=fail`) means "any of". `run` filters by `batchId`. `limit` defaults to 50; `cursor` is opaque and comes from the previous page's `nextCursor`. `total` counts every call matching the filters, not only the page. `facets` ignore the verdict filter, and `all` includes calls with no verdict, such as failed and queued ones.

`GET /api/calls/next-review?after=<callId>` → `{ "callId": string | null }`. Oldest call with open findings or verdict `needs_review`, skipping `after` and anything already human-verdicted.

`GET /api/calls/:id` → the review payload:

```json
{
  "call": CallArtifact,
  "eval": EvalResult | null,
  "summary": CallSummary,
  "audio": {
    "stereoUrl": "/api/calls/call_…/audio/recording.wav",
    "callerUrl": "/api/calls/call_…/audio/caller.wav",
    "botUrl": "/api/calls/call_…/audio/bot.wav",
    "sampleRate": 16000, "available": true
  },
  "peaks": { "perSecond": 50, "caller": [0.01, 0.42, …], "bot": [0.0, 0.0, …] } | null,
  "speech": { "caller": [[1.2, 6.8]], "bot": [[0.0, 1.0]] },
  "overlaps": [[66.9, 69.3]],
  "timing": { "replyP50S": 1.42, "replyP95S": 2.9, "overlaps": 1, "deadAirEvents": 1, "bargeStopS": 1.5 } | null,
  "evidence": { "audio": true, "transcript": true, "intendedText": true, "toolLog": true, "dbSnapshots": true, "scenario": true },
  "prevReviewId": string | null,
  "nextReviewId": string | null
}
```

- `peaks` comes from `peaks.json` (`PEAKS_PER_SECOND` in `store/artifacts.py`), values 0..1, one array per channel of equal length.
- `speech` are VAD regions per channel; `overlaps` are intervals where both are speaking. Both derived server-side so the UI never runs VAD.
- `timing` is null when `call.timingIsReal` is false or the timing evaluator was skipped.
- `evidence` is derived from `hasToolLog`, `hasIntendedText`, the presence of db snapshots, and `scenarioId`.
- `eval` is null while `status` is `running` or `evaluating`; subscribe to `/ws/calls/:id` for it to arrive.
- Audio URLs return `audio/wav` with `Accept-Ranges: bytes` so the browser can seek. Missing audio sets `available: false` and the URLs to null.

`POST /api/calls/:id/findings/:findingId` body `{ "humanVerdict": "confirmed" | "dismissed" | null, "humanNote": string }` → `Finding`. Null clears a decision. Stored with `humanAt` server-side.

`POST /api/calls/:id/verdict` body `{ "verdict": Verdict, "note": string }` → `CallSummary`. The human call verdict is stored beside the machine verdict, never overwrites it.

`POST /api/calls/:id/reevaluate` body `{ "evaluators": string[] | null }` → `{ "accepted": true }`. Null means all. Progress arrives on `/ws/calls/:id`.

`DELETE /api/calls/:id` → 204.

### `WS /ws/calls/:id`

Server → client messages while a call is running or evaluating, then the socket closes.

```json
{ "type": "status", "status": CallStatus }
{ "type": "evaluator", "evaluator": EvaluatorRun }                 // one per evaluator as it finishes or is skipped
{ "type": "finding", "finding": Finding }
{ "type": "eval", "eval": EvalResult }                              // final, once
{ "type": "error", "message": "…" }
```

## Personas

- `GET /api/personas` → `{ "items": (Persona & { "usage": { "calls": number, "counts": Counts } })[] }`
- `GET /api/personas/:id` → `Persona & { usage, recentCalls: CallSummary[] }`
- `POST /api/personas` body `Persona` without `id`, `createdAt`, `updatedAt` → `Persona` (201)
- `PUT /api/personas/:id` body `Persona` → `Persona`
- `DELETE /api/personas/:id` → 204, or 409 `conflict` if a running batch uses it
- `GET /api/voices` → `{ "items": [ { "provider": "qwen3_tts", "voiceId": "default_male_1", "label": "Default male 1", "language": "en-US" } ] }`
- `POST /api/tts/preview` body `{ "text": string, "voice": VoiceRef, "line": LineConditions | null }` → `audio/wav`. Returns 503 `model_not_ready` while TTS loads.
- `GET /api/personas/:id/prompt` → `{ "prompt": string }` the exact caller instructions rendered from the persona.
- `POST /api/personas/prompt` body as for create → `{ "prompt": string }`. The same text for a form that isn't saved yet, for the "how this reads to the caller model" panel.

## Scenarios

- `GET /api/scenarios` → `{ "items": (Scenario & { "usage": { "calls": number, "counts": Counts }, "version": number })[] }`
- `GET /api/scenarios/:id` → `Scenario & { usage, version, recentCalls: CallSummary[] }`
- `POST /api/scenarios` → 201 `Scenario`
- `PUT /api/scenarios/:id` → `Scenario`. Saving bumps `version`; existing calls keep `scenarioSnapshot`.
- `DELETE /api/scenarios/:id` → 204 or 409.
- `GET /api/customers` → `{ "items": [ { "id": "cus_1001", "name": "Dana Whitfield", "note": "Charged twice for invoice FH-4419.", "plans": ["Website Pro"], "domain": "whitfieldlandscaping.example" } ] }` from `seeds/company.yaml`.
- `GET /api/customers/:id` → `{ "customer": {...}, "subscriptions": [...], "invoices": [...], "payments": [...], "refunds": [...], "domains": [...], "mailboxes": [...], "websites": [...], "dnsRecords": [...], "tickets": [...] }` raw seed rows, snake_case keys preserved as column names.
- `POST /api/scenarios/draft` body `{ "description": string, "customerId": string | null, "baseScenarioId": string | null }` → `{ "scenario": Scenario (no id), "guessedFields": string[], "notes": string, "model": string }`. `guessedFields` names fields the model inferred rather than read from the seed (`"customerId"`, `"facts.amount"`). 503 `model_not_ready` while the LLM loads.
- `POST /api/scenarios/:id/dry-run` body `{ "personaId": string | null }` → `{ "turns": [ { "speaker": Speaker, "text": string } ], "reachedCheckpoints": string[], "model": string }`. Text only, no audio.

## Runs (batches)

- `GET /api/matrix?profile=<id>&window=7d` → `{ "cells": [ { "scenarioId": string, "personaId": string, "counts": Counts } ] }`
- `GET /api/profiles` → `{ "items": Profile[], "activeId": string }`
- `POST /api/runs` body:
  ```json
  { "pairs": [ { "scenarioId": "scn_…", "personaId": "per_…" } ], "reps": 3, "profile": "local-default",
    "mode": "auto" | "live_bots" | "turn_based", "line": "persona" | LineConditions, "concurrency": 2, "label": string | null }
  ```
  → 201 `{ "run": RunSummary, "effectiveMode": CallMode, "estimateS": number }`. `auto` picks `live_bots` when `Profile.liveCapable`, else `turn_based`.
- `GET /api/runs?cursor=&limit=` → `{ "items": RunSummary[], "nextCursor" }`
- `GET /api/runs/:id` → `{ "run": RunSummary, "calls": CallSummary[] }`
- `POST /api/runs/:id/pause` | `/resume` | `/stop` → `RunSummary`. Pause finishes the calls in flight. Stop ends them and marks them `failed` with `error: "stopped"`.
- `GET /api/runs/:id/presets`, `POST /api/presets` body `{ "name", "pairs", "reps", "profile", "mode", "line", "concurrency" }` → presets for the launcher. `GET /api/presets` lists them.

### `WS /ws/runs/:id`

```json
{ "type": "run", "run": RunSummary }                                    // on connect and whenever counts change
{ "type": "callStarted", "call": CallSummary }
{ "type": "partial", "callId": string, "speaker": Speaker, "text": string, "atS": number }      // in-progress recognition, replaces the previous partial for that speaker
{ "type": "turn", "callId": string, "utterance": Utterance }                                      // final; replaces the partial
{ "type": "event", "callId": string, "event": ToolCallEvent | BargeInEvent | SystemEvent }
{ "type": "callDone", "call": CallSummary }                             // status "evaluating" or "failed"
{ "type": "callEvaluated", "call": CallSummary }                        // verdict set
{ "type": "runDone", "run": RunSummary }
```

The client sends `{ "type": "watch", "callId": string }` to choose which concurrent call streams `partial`, `turn`, and `event`. The server defaults to the earliest running call.

## Talk to the bot

The mic call uses Pipecat's `SmallWebRTCTransport`. The browser uses `@pipecat-ai/client-js` and `@pipecat-ai/small-webrtc-transport`; the server runs `pipecat.transports.smallwebrtc`.

- `POST /api/talk/sessions` body `{ "scenarioId": string | null, "personaId": string | null, "line": LineConditions | null, "record": boolean, "profile": string | null }`
  → 201 `{ "sessionId": string, "callId": string, "offerUrl": "/api/talk/sessions/<sessionId>/offer", "iceServers": [], "sheet": { "youAre": string, "problem": string, "beats": Beat[], "mustNotSay": string[] } | null }`, or 409 `profile_busy` when bot calls fill the profile's `maxCalls`. Mic calls go ahead of queued bot calls, not ones already on the line.
  503 `model_not_ready` if the bot LLM, STT, or TTS is loading, with the role in `detail`.
- `POST /api/talk/sessions/:sessionId/offer` is the Pipecat small-WebRTC signalling endpoint. Body `{ "sdp": string, "type": "offer", "pc_id": string | null, "restart_pc": boolean }` → `{ "sdp": string, "type": "answer", "pc_id": string }`. These four keys are snake_case because Pipecat's client sends them that way; do not camelCase this one route.
- `POST /api/talk/sessions/:sessionId/end` → `{ "callId": string }`. The recording is finalised and evaluation starts; follow `/ws/calls/:callId`.
- `GET /api/talk/sessions/:sessionId` → `{ "sessionId", "callId", "state": "connecting" | "live" | "ended", "startedAt", "durationS" }`

Over the WebRTC data channel the server sends standard RTVI messages, which the client library turns into events. The dashboard relies on:

| RTVI event | Used for |
| --- | --- |
| `userStartedSpeaking` / `userStoppedSpeaking` | "you" speaking indicator |
| `botStartedSpeaking` / `botStoppedSpeaking` | bot speaking indicator |
| `userTranscript` `{ text, final, timestamp }` | your live transcript, partial then final |
| `botTranscript` `{ text }` | bot's live transcript (what it intends to say) |
| `botReady` | call connected, clock starts |
| `serverMessage` with a `voicelab` payload | everything below |

`serverMessage` payloads (`{ "voicelab": { ... } }`):

```json
{ "voicelab": { "type": "event", "event": ToolCallEvent | BargeInEvent | SystemEvent } }
{ "voicelab": { "type": "metrics", "replyP50S": 1.3, "lastReplyS": 1.1 } }
{ "voicelab": { "type": "warning", "code": "echo_suspected" | "no_headphones", "message": "…" } }
```

The client may send `{ "voicelab": { "type": "mute", "on": true } }` through `sendClientMessage`; the transport also mutes the track locally.

## Upload

- `POST /api/uploads` multipart: `file` (wav, flac, mp3, ogg), optional `scenarioId`, `botVersion`, `intendedTranscript` (text, one bot turn per line or JSON `[{ "text", "startS"? }]`), `toolLog` (text, one JSON tool call per line)
  → 201 `{ "callId": string, "probe": { "durationS": number, "channels": 1 | 2, "sampleRate": number, "channelMap": "caller_left" | "caller_right" | "mono" | "unknown", "sizeBytes": number }, "evidence": { … as in the review payload } }`
- `POST /api/uploads/:callId/channel-map` body `{ "channelMap": "caller_left" | "caller_right" | "mono" }` → `{ "ok": true }`
- `POST /api/uploads/:callId/evaluate` → `{ "accepted": true, "estimateS": number }`, then `/ws/calls/:callId`.
- `GET /api/uploads/:callId/preview?startS=0&durationS=10` → `audio/wav` for choosing the channel map.

Uploaded calls have `mode: "upload"`, `hasToolLog` and `hasIntendedText` from what was supplied, and a machine verdict that is never `pass` or `fail` when evidence is reduced (the verdict engine returns `needs_review`).

## Models and settings

- `GET /api/profiles` (above). `PUT /api/profiles/active` body `{ "id": string }` → `Profile`. Restarts only the servers whose adapter changed, and answers once they're ready, which can take as long as their `startup_timeout_s`. The selection is saved only then. Missing files give 400 `not_installed` before any server stops. Servers that don't start give 503 `profile_failed`, with the ones they replaced put back.
- `GET /api/roles` → `{ "items": [ { "role": "support_bot_llm", "label": "support bot LLM", "interface": "LLM" | "STT" | "TTS" | "RealtimeVoice" | "AudioAnalyst" | "AudioQuality", "adapter": string, "options": [ { "id": string, "label": string, "untested": boolean, "license": string } ], "health": Health } ] }`
- `PUT /api/roles/:role` body `{ "adapter": string }` → the role item. Writes the active profile YAML once the new server is up, with the same `not_installed` and `profile_failed` errors as a profile change. 409 `conflict` while a run is in progress.
- `GET /api/health` → `{ "items": Health[], "ready": boolean }`. Polled every 5 s by the shell; also `WS /ws/health` sends `{ "type": "health", "items": Health[] }` on change.
- `POST /api/servers/:role/restart` → `{ "accepted": true }`; `POST /api/servers/restart-all`; `POST /api/benchmark` → `{ "accepted": true }`, result lands in `Profile.benchmark`.
- `GET /api/system` → `{ "gpu": { "device": string, "memoryUsedGb": number, "memoryTotalGb": number }, "servers": [ { "role", "kind", "port", "pid" } ], "urls": { "local": string, "lan": string }, "cert": "self-signed" }`
- `GET /api/thresholds` → `{ "deadAirS": 2.0, "bargeStopTargetS": 0.5, "replyLatencyWarnS": 2.5, "mosBaseline": 4.0, "audioListeningMode": "advisory" }`; `PUT /api/thresholds` with the same body. `audioListeningMode` is read-only and always `advisory`.
- `GET /api/aws` → `{ "region": string | null, "credentials": "found" | "not_found", "problem": string | null, "testedAgainstLive": false }`. `problem` says why boto3 couldn't use what it found, and this makes no request. `POST /api/aws/test` → `{ "ok": boolean, "message": string }`.

## Compare

`GET /api/compare?a=<runId>&b=<runId>` →

```json
{
  "a": RunSummary, "b": RunSummary,
  "scenarioIds": [ "scn_…" ], "droppedScenarioIds": [ ],
  "criteria": [ { "criterionId": "verify_first", "text": "Verify the caller …", "kind": "must", "a": { "pass": 23, "n": 23 }, "b": { "pass": 24, "n": 24 } } ],
  "measures": [
    { "key": "replyP50S", "label": "Bot reply latency p50", "a": 1.42, "b": null, "comparable": false, "reason": "turn-based", "lowerIsBetter": true },
    { "key": "criticalMispronunciations", "label": "Critical token mispronunciations", "a": 3, "b": 1, "comparable": true, "lowerIsBetter": true },
    { "key": "missingStateChanges", "label": "Expected db changes missing", "a": 4, "b": 1, "comparable": true, "lowerIsBetter": true },
    { "key": "mos", "label": "MOS, bot channel", "a": 3.9, "b": 4.1, "comparable": true, "lowerIsBetter": false },
    { "key": "advisoryNotes", "label": "Advisory listener notes", "a": 6, "b": 4, "comparable": false, "reason": "advisory" },
    { "key": "invalid", "label": "Invalid runs", "a": 1, "b": 0, "comparable": false, "reason": "test, not bot" }
  ],
  "cells": [ { "scenarioId": "scn_…", "personaId": "per_…", "a": { "pass": 2, "n": 3 }, "b": { "pass": 3, "n": 3 } } ],
  "worst": { "criterionId": "offers_ticket", "deltaPts": -21, "callIds": [ "call_…" ] } | null
}
```

Criteria are matched across runs by `criterionId` from the scenario snapshot; criteria present in only one run are listed with the other side's `n: 0`.

## Mapping notes

Where the API does not map one-to-one onto `models.py`:

- **Open findings.** `CallSummary.openFindings` counts findings with `advisory == false` and `humanVerdict == null`. Advisory findings are listed on the review screen so they can be dismissed, but they never count as open in lists or the overview.
- **Human call verdict** has no field on `EvalResult`. Store it with the call (suggested: `human_verdict`, `human_note`, `human_at` on `CallArtifact`, or a sidecar `review.json`) and serve it as `CallSummary.humanVerdict`.
- **Invalid reason.** `CallSummary.invalidReason` is the `caller_validity` evaluator's `reason` (or its first finding's `title`) when `verdict == "invalid"`.
- **Rep number.** `CallSummary.rep` is the call's index among calls with the same `batchId`, `scenarioId`, `personaId`, 1-based.
- **Scenario version.** Not in `Scenario`; the store keeps a counter per scenario id and the API adds `version`.
- **Reply latency and overlaps** on `CallSummary` come from the timing evaluator's `metrics` (`reply_p50_s`, `overlaps`) and are null when it was skipped.
- **Speech regions and overlaps** in the review payload come from `eval/vad.py` on the stored channels and are cached beside `peaks.json`.
- **Persona interruption scale.** The UI's 1..5 "interruption tendency" is `interruptionRate` 0, 0.25, 0.5, 0.75, 1.0. Round to the nearest step when displaying.
- **Mode labels.** The UI shows `live_bots` as "live", `turn_based` as "turn-based", `human` as "mic", `upload` as "upload". The API never sends UI labels.
- **The `/offer` route** keeps Pipecat's snake_case body. Everything else is camelCase.
