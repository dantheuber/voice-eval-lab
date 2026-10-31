/* Typed endpoint functions. Every screen goes through here; nothing else calls fetch. */
import { api, qs } from "./transport";
import type * as T from "./types";

export const endpoints = {
  overview: (profile?: string, window = "24h") => api.get<T.Overview>(`/api/overview${qs({ profile, window })}`),

  calls: (q: T.CallsQuery) => api.get<T.CallsPage>(`/api/calls${qs({ ...q, openOnly: q.openOnly ? "true" : undefined })}`),
  nextReview: (after?: string) => api.get<{ callId: string | null }>(`/api/calls/next-review${qs({ after })}`),
  call: (id: string) => api.get<T.ReviewPayload>(`/api/calls/${id}`),
  decideFinding: (callId: string, findingId: string, body: { humanVerdict: T.HumanVerdict | null; humanNote: string }) =>
    api.post<T.Finding>(`/api/calls/${callId}/findings/${findingId}`, body),
  setVerdict: (callId: string, body: { verdict: T.Verdict; note: string }) => api.post<T.CallSummary>(`/api/calls/${callId}/verdict`, body),
  reevaluate: (callId: string, evaluators: string[] | null) => api.post<{ accepted: true }>(`/api/calls/${callId}/reevaluate`, { evaluators }),
  deleteCall: (callId: string) => api.del(`/api/calls/${callId}`),
  callSocket: (callId: string) => api.socket(`/ws/calls/${callId}`),

  personas: () => api.get<{ items: T.PersonaListItem[] }>("/api/personas"),
  persona: (id: string) => api.get<T.PersonaDetail>(`/api/personas/${id}`),
  createPersona: (body: T.PersonaInput) => api.post<T.Persona>("/api/personas", body),
  updatePersona: (id: string, body: T.Persona) => api.put<T.Persona>(`/api/personas/${id}`, body),
  deletePersona: (id: string) => api.del(`/api/personas/${id}`),
  personaPrompt: (id: string) => api.get<{ prompt: string }>(`/api/personas/${id}/prompt`),
  personaPromptPreview: (body: T.PersonaInput) => api.post<{ prompt: string }>("/api/personas/prompt", body),
  voices: () => api.get<{ items: T.Voice[] }>("/api/voices"),
  ttsPreview: (body: { text: string; voice: T.VoiceRef; line: T.LineConditions | null }) => api.blob("/api/tts/preview", "POST", body),

  scenarios: () => api.get<{ items: T.ScenarioListItem[] }>("/api/scenarios"),
  scenario: (id: string) => api.get<T.ScenarioDetail>(`/api/scenarios/${id}`),
  createScenario: (body: T.ScenarioInput) => api.post<T.Scenario>("/api/scenarios", body),
  updateScenario: (id: string, body: T.Scenario) => api.put<T.Scenario>(`/api/scenarios/${id}`, body),
  deleteScenario: (id: string) => api.del(`/api/scenarios/${id}`),
  customers: () => api.get<{ items: T.Customer[] }>("/api/customers"),
  customer: (id: string) => api.get<Record<string, unknown>>(`/api/customers/${id}`),
  draftScenario: (body: { description: string; customerId: string | null; baseScenarioId: string | null }) => api.post<T.ScenarioDraft>("/api/scenarios/draft", body),
  dryRun: (id: string, personaId: string | null) => api.post<T.DryRun>(`/api/scenarios/${id}/dry-run`, { personaId }),

  matrix: (profile?: string, window = "7d") => api.get<{ cells: T.MatrixCell[] }>(`/api/matrix${qs({ profile, window })}`),
  profiles: () => api.get<{ items: T.Profile[]; activeId: string }>("/api/profiles"),
  setActiveProfile: (id: string) => api.put<T.Profile>("/api/profiles/active", { id }),
  createRun: (body: T.RunRequest) => api.post<T.RunCreated>("/api/runs", body),
  runs: () => api.get<{ items: T.RunSummary[]; nextCursor: string | null }>("/api/runs"),
  run: (id: string) => api.get<T.RunDetail>(`/api/runs/${id}`),
  runAction: (id: string, action: "pause" | "resume" | "stop") => api.post<T.RunSummary>(`/api/runs/${id}/${action}`),
  runSocket: (id: string) => api.socket(`/ws/runs/${id}`),

  createTalkSession: (body: { scenarioId: string | null; personaId: string | null; line: T.LineConditions | null; record: boolean; profile: string | null }) =>
    api.post<T.TalkSession>("/api/talk/sessions", body),
  endTalkSession: (sessionId: string) => api.post<{ callId: string }>(`/api/talk/sessions/${sessionId}/end`),
  talkSession: (sessionId: string) => api.get<T.TalkSessionState>(`/api/talk/sessions/${sessionId}`),

  upload: (form: FormData) => api.form<T.UploadResult>("/api/uploads", form),
  uploadChannelMap: (callId: string, channelMap: string) => api.post<{ ok: true }>(`/api/uploads/${callId}/channel-map`, { channelMap }),
  evaluateUpload: (callId: string) => api.post<{ accepted: true; estimateS: number }>(`/api/uploads/${callId}/evaluate`),

  roles: () => api.get<{ items: T.RoleItem[] }>("/api/roles"),
  setRole: (role: string, adapter: string) => api.put<T.RoleItem>(`/api/roles/${role}`, { adapter }),
  health: () => api.get<{ items: T.Health[]; ready: boolean }>("/api/health"),
  healthSocket: () => api.socket("/ws/health"),
  restartServer: (role: string) => api.post<{ accepted: true }>(`/api/servers/${role}/restart`),
  restartAll: () => api.post<{ accepted: true }>("/api/servers/restart-all"),
  benchmark: () => api.post<{ accepted: true }>("/api/benchmark"),
  system: () => api.get<T.SystemInfo>("/api/system"),
  thresholds: () => api.get<T.Thresholds>("/api/thresholds"),
  setThresholds: (body: T.Thresholds) => api.put<T.Thresholds>("/api/thresholds", body),
  aws: () => api.get<T.AwsStatus>("/api/aws"),
  awsTest: () => api.post<{ ok: boolean; message: string }>("/api/aws/test"),

  compare: (a: string, b: string) => api.get<T.Compare>(`/api/compare${qs({ a, b })}`),

  mediaUrl: (p: string | null) => api.mediaUrl(p),
};
