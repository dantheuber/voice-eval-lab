/* In-browser mock of docs/API.md. Stateful for the session so saves, decisions, and launches stick. */
import { ApiError, type Method, type Socket, type Transport } from "../transport";
import type * as T from "../types";
import * as FX from "./fixtures";
import { synthStereoWav } from "./audio";

type Handler = (m: RegExpMatchArray, body: unknown, url: URL) => unknown;
type Route = { method: Method; re: RegExp; fn: Handler };

const LATENCY = () => 120 + Math.random() * 180;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clone = <X>(x: X): X => JSON.parse(JSON.stringify(x)) as X;
const notFound = (what: string) => new ApiError(404, { code: "not_found", message: `${what} not found` }, "not found");
const newId = (p: string) => `${p}_${Math.random().toString(16).slice(2, 14)}`;

/* Mutations survive a reload within the tab, so navigating away and back keeps decisions and saved forms. */
interface Persisted {
  personas: T.Persona[]; scenarios: T.Scenario[]; versions: Record<string, number>; activeProfile: string; thresholds: T.Thresholds;
  decisions: Record<string, Record<string, { humanVerdict: T.HumanVerdict | null; humanNote: string; humanAt: string | null }>>;
  verdicts: Record<string, T.Verdict>;
}
const STORE_KEY = "vl-mock-state";
function loadPersisted(): Partial<Persisted> {
  try { const raw = sessionStorage.getItem(STORE_KEY); return raw ? (JSON.parse(raw) as Partial<Persisted>) : {}; } catch { return {}; }
}

export function createMockTransport(): Transport {
  // ---- state ----
  const saved = loadPersisted();
  const personas = saved.personas ?? clone(FX.personas);
  const scenarios = saved.scenarios ?? clone(FX.scenarios);
  const versions: Record<string, number> = saved.versions ?? Object.fromEntries(scenarios.map((s) => [s.id, 3]));
  const runs = clone(FX.runs);
  const calls = clone(FX.calls);
  const reviews: Record<string, T.ReviewPayload> = { [FX.reviewA.call.id]: clone(FX.reviewA), [FX.reviewB.call.id]: clone(FX.reviewB) };
  const health = clone(FX.health);
  const roles = clone(FX.roles);
  const profiles = clone(FX.profiles);
  let activeProfile = saved.activeProfile ?? "local-default";
  let thresholds = saved.thresholds ?? clone(FX.thresholds);
  const decisions = saved.decisions ?? {};
  const verdicts = saved.verdicts ?? {};
  const persist = () => { try { sessionStorage.setItem(STORE_KEY, JSON.stringify({ personas, scenarios, versions, activeProfile, thresholds, decisions, verdicts } satisfies Persisted)); } catch { /* quota or private mode */ } };
  const applyDecisions = (r: T.ReviewPayload) => {
    const d = decisions[r.call.id];
    if (d && r.eval) r.eval.findings.forEach((f) => { const x = d[f.id]; if (x) Object.assign(f, x); });
    const v = verdicts[r.call.id]; if (v) r.summary.humanVerdict = v;
  };
  calls.forEach((c) => { const v = verdicts[c.id]; if (v) c.humanVerdict = v; });
  Object.values(reviews).forEach(applyDecisions);
  const runCalls: Record<string, T.CallSummary[]> = {};
  const audioUrls: Record<string, string> = {};
  const presets: unknown[] = [];
  const talkSessions: Record<string, { callId: string; startedAt: number; scenarioId: string | null }> = {};
  const healthListeners = new Set<(items: T.Health[]) => void>();

  // judge LLM finishes loading after 25 s of session time
  const judge = health.find((h) => h.role === "llm_judge")!;
  const t0 = Date.now();
  const tickHealth = () => {
    const el = Math.round((Date.now() - t0) / 1000) + 22;
    if (judge.state === "loading") {
      if (el >= 47) { judge.state = "ok"; judge.detail = "ready, 21 tok/s"; judge.elapsedS = null; judge.usualS = null; healthListeners.forEach((f) => f(clone(health))); }
      else { judge.elapsedS = el; }
    }
    roles.forEach((r) => { const h = health.find((x) => x.role === r.role); if (h) r.health = clone(h); });
  };
  setInterval(tickHealth, 1000);

  const review = (id: string): T.ReviewPayload => {
    if (!reviews[id]) {
      const s = calls.find((c) => c.id === id) ?? Object.values(runCalls).flat().find((c) => c.id === id);
      if (!s) throw notFound("call");
      reviews[id] = FX.genericReview(s);
      applyDecisions(reviews[id]);
      recount(id);
    }
    return reviews[id];
  };
  const summaryOf = (id: string) => calls.find((c) => c.id === id) ?? Object.values(runCalls).flat().find((c) => c.id === id);
  const recount = (id: string) => {
    const r = reviews[id]; const s = summaryOf(id);
    if (!r?.eval || !s) return;
    s.findings = r.eval.findings.length;
    s.openFindings = r.eval.findings.filter((f) => !f.advisory && !f.humanVerdict).length;
    r.summary = s;
  };
  const allCalls = () => [...calls, ...Object.values(runCalls).flat()];
  const usage = (pred: (c: T.CallSummary) => boolean): T.Usage => {
    const cs = allCalls().filter(pred);
    const counts: T.Counts = { pass: 0, fail: 0, needsReview: 0, invalid: 0 };
    cs.forEach((c) => { if (c.verdict === "pass") counts.pass++; else if (c.verdict === "fail") counts.fail++; else if (c.verdict === "needs_review") counts.needsReview++; else if (c.verdict === "invalid") counts.invalid++; });
    return { calls: cs.length, counts };
  };
  const queueIds = () => allCalls().filter((c) => c.status === "done" && !c.humanVerdict && (c.openFindings > 0 || c.verdict === "needs_review")).sort((a, b) => (a.startedAt ?? "").localeCompare(b.startedAt ?? ""));

  const routes: Route[] = [
    { method: "GET", re: /^\/api\/overview$/, fn: () => {
      const q = queueIds();
      const ov: T.Overview = {
        window: "24h", profile: activeProfile, counts: runs.slice(0, 5).reduce((a, r) => ({ pass: a.pass + r.counts.pass, fail: a.fail + r.counts.fail, needsReview: a.needsReview + r.counts.needsReview, invalid: a.invalid + r.counts.invalid }), { pass: 0, fail: 0, needsReview: 0, invalid: 0 }),
        queue: q.slice(0, 8).map((c) => FX.queue.find((x) => x.callId === c.id) ?? { callId: c.id, scenarioId: c.scenarioId, scenarioTitle: c.scenarioTitle, personaName: c.personaName, openFindings: c.openFindings, why: `${c.openFindings} open findings`, createdAt: c.startedAt ?? FX.iso(0) }),
        queueTotal: q.length, openFindingsTotal: q.reduce((a, c) => a + c.openFindings, 0), reviewPaceSPerFinding: 75,
        callsPerHour: [0, 0, 0, 0, 0, 0, 0, 1, 4, 9, 12, 6, 2, 0, 0, 2, 8, 10, 5, 3, 1, 0, 0, 0], recentRuns: runs.slice(0, 6), runningRuns: runs.filter((r) => r.status === "running").length,
        scenarioRates: FX.scenarioRates, health: clone(health), topOpenFindings: FX.topOpenFindings,
      };
      return ov;
    } },
    { method: "GET", re: /^\/api\/calls$/, fn: (_m, _b, url) => {
      const p = url.searchParams;
      const verdicts = p.getAll("verdict");
      let items = allCalls().filter((c) =>
        (!verdicts.length || (c.verdict && verdicts.includes(c.verdict))) && (!p.get("scenario") || c.scenarioId === p.get("scenario")) && (!p.get("persona") || c.personaId === p.get("persona"))
        && (!p.get("mode") || c.mode === p.get("mode")) && (!p.get("profile") || c.profile === p.get("profile")) && (!p.get("run") || c.batchId === p.get("run")) && (p.get("openOnly") !== "true" || c.openFindings > 0) && (!p.get("status") || c.status === p.get("status")));
      items = items.sort((a, b) => (b.startedAt ?? "").localeCompare(a.startedAt ?? ""));
      const facets: Record<string, number> = {};
      allCalls().forEach((c) => { if (c.verdict) facets[c.verdict] = (facets[c.verdict] ?? 0) + 1; });
      // opaque cursor = offset, limit defaults to 50 as in docs/API.md (`?limit=` lets tests exercise paging)
      const limit = Math.max(1, Number(p.get("limit")) || 50);
      const offset = Math.max(0, Number(p.get("cursor")) || 0);
      const slice = items.slice(offset, offset + limit);
      const page: T.CallsPage = { items: slice, nextCursor: offset + limit < items.length ? String(offset + limit) : null, total: items.length, facets: { verdict: facets } };
      return page;
    } },
    { method: "GET", re: /^\/api\/calls\/next-review$/, fn: (_m, _b, url) => {
      const after = url.searchParams.get("after");
      const q = queueIds().map((c) => c.id);
      const i = after ? q.indexOf(after) : -1;
      return { callId: q.find((id, j) => j > i && id !== after) ?? q.find((id) => id !== after) ?? null };
    } },
    { method: "GET", re: /^\/api\/calls\/([^/]+)$/, fn: (m) => {
      const r = review(m[1]);
      const q = queueIds().map((c) => c.id); const i = q.indexOf(m[1]);
      r.nextReviewId = q[i + 1] ?? q.find((x) => x !== m[1]) ?? null; r.prevReviewId = i > 0 ? q[i - 1] : null;
      return r;
    } },
    { method: "POST", re: /^\/api\/calls\/([^/]+)\/findings\/([^/]+)$/, fn: (m, body) => {
      const r = review(m[1]); const b = body as { humanVerdict: T.HumanVerdict | null; humanNote: string };
      const f = r.eval?.findings.find((x) => x.id === m[2]); if (!f) throw notFound("finding");
      f.humanVerdict = b.humanVerdict; f.humanNote = b.humanNote ?? f.humanNote; f.humanAt = b.humanVerdict ? new Date().toISOString() : null;
      (decisions[m[1]] ??= {})[f.id] = { humanVerdict: f.humanVerdict, humanNote: f.humanNote, humanAt: f.humanAt };
      recount(m[1]); persist(); return f;
    } },
    { method: "POST", re: /^\/api\/calls\/([^/]+)\/verdict$/, fn: (m, body) => { const s = summaryOf(m[1]); if (!s) throw notFound("call"); s.humanVerdict = (body as { verdict: T.Verdict }).verdict; verdicts[m[1]] = s.humanVerdict; persist(); return s; } },
    { method: "POST", re: /^\/api\/calls\/([^/]+)\/reevaluate$/, fn: () => ({ accepted: true }) },
    { method: "DELETE", re: /^\/api\/calls\/([^/]+)$/, fn: (m) => { const i = calls.findIndex((c) => c.id === m[1]); if (i >= 0) calls.splice(i, 1); return undefined; } },

    { method: "GET", re: /^\/api\/personas$/, fn: () => ({ items: personas.map((p) => ({ ...p, usage: usage((c) => c.personaId === p.id) })) }) },
    { method: "POST", re: /^\/api\/personas\/prompt$/, fn: (_m, body) => ({ prompt: renderPrompt(body as T.Persona) }) },
    { method: "GET", re: /^\/api\/personas\/([^/]+)\/prompt$/, fn: (m) => { const p = personas.find((x) => x.id === m[1]); if (!p) throw notFound("persona"); return { prompt: renderPrompt(p) }; } },
    { method: "GET", re: /^\/api\/personas\/([^/]+)$/, fn: (m) => { const p = personas.find((x) => x.id === m[1]); if (!p) throw notFound("persona"); return { ...p, usage: usage((c) => c.personaId === p.id), recentCalls: allCalls().filter((c) => c.personaId === p.id).slice(0, 8) }; } },
    { method: "POST", re: /^\/api\/personas$/, fn: (_m, body) => { const p = { ...(body as T.PersonaInput), id: newId("per"), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() }; personas.unshift(p); persist(); return p; } },
    { method: "PUT", re: /^\/api\/personas\/([^/]+)$/, fn: (m, body) => { const i = personas.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound("persona"); personas[i] = { ...(body as T.Persona), id: m[1], updatedAt: new Date().toISOString() }; persist(); return personas[i]; } },
    { method: "DELETE", re: /^\/api\/personas\/([^/]+)$/, fn: (m) => { const i = personas.findIndex((x) => x.id === m[1]); if (i >= 0) personas.splice(i, 1); persist(); return undefined; } },
    { method: "GET", re: /^\/api\/voices$/, fn: () => ({ items: FX.voices }) },

    { method: "GET", re: /^\/api\/scenarios$/, fn: () => ({ items: scenarios.map((s) => ({ ...s, version: versions[s.id] ?? 1, usage: usage((c) => c.scenarioId === s.id) })) }) },
    { method: "GET", re: /^\/api\/scenarios\/([^/]+)$/, fn: (m) => { const s = scenarios.find((x) => x.id === m[1]); if (!s) throw notFound("scenario"); return { ...s, version: versions[s.id] ?? 1, usage: usage((c) => c.scenarioId === s.id), recentCalls: allCalls().filter((c) => c.scenarioId === s.id).slice(0, 8) }; } },
    { method: "POST", re: /^\/api\/scenarios$/, fn: (_m, body) => { const s = { ...(body as T.ScenarioInput), id: newId("scn"), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() }; scenarios.unshift(s); versions[s.id] = 1; persist(); return s; } },
    { method: "PUT", re: /^\/api\/scenarios\/([^/]+)$/, fn: (m, body) => { const i = scenarios.findIndex((x) => x.id === m[1]); if (i < 0) throw notFound("scenario"); scenarios[i] = { ...(body as T.Scenario), id: m[1], updatedAt: new Date().toISOString() }; versions[m[1]] = (versions[m[1]] ?? 1) + 1; persist(); return scenarios[i]; } },
    { method: "DELETE", re: /^\/api\/scenarios\/([^/]+)$/, fn: (m) => { const i = scenarios.findIndex((x) => x.id === m[1]); if (i >= 0) scenarios.splice(i, 1); persist(); return undefined; } },
    { method: "POST", re: /^\/api\/scenarios\/draft$/, fn: async (_m, body) => { await sleep(1400); return draft(body as { description: string; customerId: string | null }); } },
    { method: "POST", re: /^\/api\/scenarios\/([^/]+)\/dry-run$/, fn: async (m) => {
      await sleep(900); const s = scenarios.find((x) => x.id === m[1]); if (!s) throw notFound("scenario");
      const dr: T.DryRun = { model: "Qwen3.5-9B", reachedCheckpoints: s.beats.map((b) => b.id), turns: [
        { speaker: "bot", text: "Thanks for calling Fernhill Internet, this is Ada. How can I help?" }, { speaker: "caller", text: s.goal },
        { speaker: "bot", text: "I can help with that. Can I get the billing zip and PIN on the account?" }, { speaker: "caller", text: `${s.facts.billing_zip ?? "97214"}, and the PIN is ${s.facts.pin ?? "4821"}.` },
        { speaker: "bot", text: "Thanks. One moment while I check." }, { speaker: "caller", text: "Sure." },
      ] };
      return dr;
    } },
    { method: "GET", re: /^\/api\/customers$/, fn: () => ({ items: FX.customers }) },
    { method: "GET", re: /^\/api\/customers\/([^/]+)$/, fn: (m) => { const c = FX.customers.find((x) => x.id === m[1]); if (!c) throw notFound("customer"); return { customer: c, invoices: [{ id: "FH-4419", amount_cents: 4999, status: "paid" }], refunds: [], domains: [{ name: c.domain }] }; } },

    { method: "GET", re: /^\/api\/matrix$/, fn: () => ({ cells: FX.matrix }) },
    { method: "GET", re: /^\/api\/profiles$/, fn: () => ({ items: profiles, activeId: activeProfile }) },
    { method: "PUT", re: /^\/api\/profiles\/active$/, fn: (_m, body) => { activeProfile = (body as { id: string }).id; persist(); return profiles.find((p) => p.id === activeProfile); } },
    { method: "POST", re: /^\/api\/runs$/, fn: (_m, body) => {
      const b = body as T.RunRequest;
      if (!b.pairs.length) throw new ApiError(400, { code: "invalid", message: "Select at least one scenario and persona pair." }, "invalid");
      const prof = profiles.find((p) => p.id === b.profile);
      const effectiveMode: T.CallMode = b.mode === "auto" ? (prof?.liveCapable ? "live_bots" : "turn_based") : b.mode;
      const id = `batch_${String(143 + Object.keys(runCalls).length).padStart(4, "0")}`;
      const total = b.pairs.length * b.reps;
      const run: T.RunSummary = { id, label: b.label ?? `${new Set(b.pairs.map((p) => p.scenarioId)).size} scenarios x ${new Set(b.pairs.map((p) => p.personaId)).size} personas x ${b.reps}`, profile: b.profile, mode: effectiveMode, status: "running", startedAt: new Date().toISOString(), endedAt: null, calls: total, done: 0, running: 0, counts: { pass: 0, fail: 0, needsReview: 0, invalid: 0 }, etaS: total * (effectiveMode === "live_bots" ? 105 : 210), scenarioIds: [...new Set(b.pairs.map((p) => p.scenarioId))], personaIds: [...new Set(b.pairs.map((p) => p.personaId))], reps: b.reps };
      runs.unshift(run);
      runCalls[id] = [];
      for (let rep = 1; rep <= b.reps; rep++) for (const pr of b.pairs) {
        const sc = scenarios.find((s) => s.id === pr.scenarioId); const pe = personas.find((p) => p.id === pr.personaId);
        runCalls[id].push({ id: newId("call"), batchId: id, rep, scenarioId: pr.scenarioId, scenarioTitle: sc?.title ?? null, personaId: pr.personaId, personaName: pe?.name ?? null, mode: effectiveMode, status: "queued", profile: b.profile, verdict: null, humanVerdict: null, invalidReason: null, findings: 0, openFindings: 0, durationS: 0, replyP50S: null, overlaps: null, startedAt: null, timingIsReal: effectiveMode === "live_bots" });
      }
      const created: T.RunCreated = { run, effectiveMode, estimateS: run.etaS ?? 0 };
      return created;
    } },
    { method: "GET", re: /^\/api\/runs$/, fn: () => ({ items: runs, nextCursor: null }) },
    { method: "GET", re: /^\/api\/runs\/([^/]+)$/, fn: (m) => { const run = runs.find((r) => r.id === m[1]); if (!run) throw notFound("run"); return { run, calls: runCalls[m[1]] ?? calls.filter((c) => c.batchId === m[1]) }; } },
    { method: "POST", re: /^\/api\/runs\/([^/]+)\/(pause|resume|stop)$/, fn: (m) => { const run = runs.find((r) => r.id === m[1]); if (!run) throw notFound("run"); run.status = m[2] === "pause" ? "paused" : m[2] === "resume" ? "running" : "stopped"; if (m[2] === "stop") run.endedAt = new Date().toISOString(); return run; } },
    { method: "GET", re: /^\/api\/presets$/, fn: () => ({ items: presets }) },
    { method: "POST", re: /^\/api\/presets$/, fn: (_m, body) => { presets.push(body); return body; } },

    { method: "POST", re: /^\/api\/talk\/sessions$/, fn: (_m, body) => {
      const b = body as { scenarioId: string | null };
      const sessionId = newId("talk"); const callId = newId("call");
      talkSessions[sessionId] = { callId, startedAt: Date.now(), scenarioId: b.scenarioId };
      const sc = scenarios.find((s) => s.id === b.scenarioId) ?? null;
      const s: T.TalkSession = { sessionId, callId, offerUrl: `/api/talk/sessions/${sessionId}/offer`, iceServers: [{ urls: "stun:stun.l.google.com:19302" }], sheet: sc ? { youAre: `${sc.facts.name ?? "the customer"}, ${FX.customers.find((c) => c.id === sc.customerId)?.domain ?? ""}, billing zip ${sc.facts.billing_zip ?? "?"}, PIN ${sc.facts.pin ?? "?"}`, problem: sc.goal, beats: sc.beats, mustNotSay: ["your full card number", "anything about a different invoice"] } : null };
      return s;
    } },
    { method: "POST", re: /^\/api\/talk\/sessions\/([^/]+)\/end$/, fn: (m) => {
      const s = talkSessions[m[1]]; if (!s) throw notFound("session");
      const sc = scenarios.find((x) => x.id === s.scenarioId);
      const dur = Math.max(20, Math.round((Date.now() - s.startedAt) / 1000));
      const summary: T.CallSummary = { id: s.callId, batchId: "batch_0139", rep: 1, scenarioId: sc?.id ?? null, scenarioTitle: sc?.title ?? null, personaId: null, personaName: "you", mode: "human", status: "evaluating", profile: activeProfile, verdict: null, humanVerdict: null, invalidReason: null, findings: 0, openFindings: 0, durationS: dur, replyP50S: 1.3, overlaps: 1, startedAt: new Date(s.startedAt).toISOString(), timingIsReal: true };
      calls.unshift(summary);
      return { callId: s.callId };
    } },
    { method: "GET", re: /^\/api\/talk\/sessions\/([^/]+)$/, fn: (m) => { const s = talkSessions[m[1]]; if (!s) throw notFound("session"); return { sessionId: m[1], callId: s.callId, state: "live", startedAt: new Date(s.startedAt).toISOString(), durationS: (Date.now() - s.startedAt) / 1000 }; } },

    { method: "POST", re: /^\/api\/uploads$/, fn: async (_m, body) => {
      await sleep(600);
      const form = body as FormData; const file = form.get("file") as File | null; const scenarioId = (form.get("scenarioId") as string) || null;
      const callId = newId("call");
      const sc = scenarios.find((s) => s.id === scenarioId);
      calls.unshift({ id: callId, batchId: "batch_0137", rep: 1, scenarioId, scenarioTitle: sc?.title ?? null, personaId: null, personaName: null, mode: "upload", status: "evaluating", profile: activeProfile, verdict: null, humanVerdict: null, invalidReason: null, findings: 0, openFindings: 0, durationS: 311, replyP50S: 1.9, overlaps: 1, startedAt: new Date().toISOString(), timingIsReal: true });
      const hasIntended = !!form.get("intendedTranscript"); const hasTools = !!form.get("toolLog");
      const res: T.UploadResult = { callId, probe: { durationS: 311, channels: 2, sampleRate: 8000, channelMap: "caller_left", sizeBytes: file?.size ?? 4_980_000 }, evidence: { audio: true, transcript: true, intendedText: hasIntended, toolLog: hasTools, dbSnapshots: false, scenario: !!scenarioId } };
      return res;
    } },
    { method: "POST", re: /^\/api\/uploads\/([^/]+)\/channel-map$/, fn: () => ({ ok: true }) },
    { method: "POST", re: /^\/api\/uploads\/([^/]+)\/evaluate$/, fn: () => ({ accepted: true, estimateS: 40 }) },

    { method: "GET", re: /^\/api\/roles$/, fn: () => { tickHealth(); return { items: roles }; } },
    { method: "PUT", re: /^\/api\/roles\/([^/]+)$/, fn: (m, body) => { const r = roles.find((x) => x.role === m[1]); if (!r) throw notFound("role"); r.adapter = (body as { adapter: string }).adapter; r.health = { ...r.health, state: "loading", detail: "restarting", elapsedS: 1, usualS: 30 }; setTimeout(() => { r.health = { ...r.health, state: "ok", detail: "ready", elapsedS: null, usualS: null }; }, 6000); return r; } },
    { method: "GET", re: /^\/api\/health$/, fn: () => { tickHealth(); return { items: clone(health), ready: health.every((h) => h.state === "ok" || h.state === "off") }; } },
    { method: "POST", re: /^\/api\/servers\/([^/]+)\/restart$/, fn: () => ({ accepted: true }) },
    { method: "POST", re: /^\/api\/servers\/restart-all$/, fn: () => ({ accepted: true }) },
    { method: "POST", re: /^\/api\/benchmark$/, fn: () => ({ accepted: true }) },
    { method: "GET", re: /^\/api\/system$/, fn: () => FX.system },
    { method: "GET", re: /^\/api\/thresholds$/, fn: () => thresholds },
    { method: "PUT", re: /^\/api\/thresholds$/, fn: (_m, body) => { thresholds = { ...(body as T.Thresholds), audioListeningMode: "advisory" }; persist(); return thresholds; } },
    { method: "GET", re: /^\/api\/aws$/, fn: () => ({ region: "us-west-2", credentials: "not_found", testedAgainstLive: false }) },
    { method: "POST", re: /^\/api\/aws\/test$/, fn: () => ({ ok: false, message: "No AWS credentials found. Set AWS_PROFILE or keys in .env." }) },

    { method: "GET", re: /^\/api\/compare$/, fn: (_m, _b, url) => {
      const a = runs.find((r) => r.id === url.searchParams.get("a")); const b = runs.find((r) => r.id === url.searchParams.get("b"));
      if (!a || !b) throw notFound("run");
      return FX.compareFixture(a, b);
    } },
  ];

  async function request<X>(method: Method, path: string, body?: unknown, init?: { form?: FormData }): Promise<X> {
    await sleep(LATENCY());
    const url = new URL(path, "http://mock.local");
    for (const r of routes) {
      if (r.method !== method) continue;
      const m = url.pathname.match(r.re);
      if (m) return clone(await r.fn(m, init?.form ?? body, url)) as X;
    }
    throw notFound(`${method} ${url.pathname}`);
  }

  function mediaUrl(p: string | null): string | null {
    if (!p) return null;
    const m = p.match(/^\/api\/calls\/([^/]+)\/audio\//);
    if (!m) return p;
    const id = m[1];
    if (!audioUrls[id]) {
      const r = review(id);
      audioUrls[id] = URL.createObjectURL(synthStereoWav(r.speech.caller, r.speech.bot, r.call.durationS));
    }
    return audioUrls[id];
  }

  async function blob(path: string): Promise<Blob> {
    await sleep(900);
    if (path.startsWith("/api/tts/preview")) return synthStereoWav([[0.2, 3.4]], [], 3.6);
    throw notFound(path);
  }

  function socket(path: string): Socket {
    const s = makeSocket();
    const runM = path.match(/^\/ws\/runs\/([^/]+)$/);
    const callM = path.match(/^\/ws\/calls\/([^/]+)$/);
    if (runM) scriptRun(runM[1], s);
    else if (callM) scriptCall(callM[1], s);
    else if (path === "/ws/health") { const f = (items: T.Health[]) => s.emit({ type: "health", items }); healthListeners.add(f); s.onCloseInternal = () => healthListeners.delete(f); }
    else setTimeout(() => s.close(), 10);
    return s;
  }

  function scriptRun(id: string, s: MockSocket) {
    const run = runs.find((r) => r.id === id);
    if (!run) { setTimeout(() => s.close(), 10); return; }
    const list = runCalls[id] ?? calls.filter((c) => c.batchId === id);
    s.emit({ type: "run", run });
    if (run.status !== "running") return;
    let stopped = false;
    s.onCloseInternal = () => { stopped = true; };
    const script: [T.Speaker | "tool" | "barge", string][] = [
      ["bot", "Thanks for calling Fernhill Internet, this is Ada. How can I help?"],
      ["caller", "Hi, my website's down. It says suspended. I paid that invoice on Tuesday."],
      ["tool", "lookup_customer"],
      ["bot", "Sorry to hear that. Can I get the billing zip and PIN on the account?"],
      ["caller", "Nine seven zero three five. Seven three one six."],
      ["tool", "get_site_status"],
      ["bot", "Thanks, Marcus. I can see bellhardware dot example was suspended for invoice FH-4502, and that payment did arrive on Tuesday. Let me reinstate it now."],
      ["caller", "Right, well it hasn't come back, and I've had customers calling me about it since"],
      ["barge", ""],
    ];
    const concurrency = 2;
    const driveCall = async (c: T.CallSummary) => {
      if (stopped) return;
      c.status = "running"; c.startedAt = new Date().toISOString(); run.running++;
      s.emit({ type: "callStarted", call: c }); s.emit({ type: "run", run });
      let t = 0;
      for (const [who, text] of script) {
        if (stopped || run.status !== "running") return;
        if (who === "tool") { await sleep(500); t += 1; s.emit({ type: "event", callId: c.id, event: { type: "tool_call", id: newId("tc"), atS: t, name: text, arguments: {}, result: "ok", error: null, durationMs: 90 } }); continue; }
        if (who === "barge") { s.emit({ type: "event", callId: c.id, event: { type: "barge_in", id: newId("bi"), atS: t, botStoppedAtS: t + 0.3, interruptedUtteranceId: null } }); continue; }
        for (let i = 3; i < text.length; i += 4) { if (stopped) return; s.emit({ type: "partial", callId: c.id, speaker: who, text: text.slice(0, i), atS: t }); await sleep(45); }
        s.emit({ type: "turn", callId: c.id, utterance: { type: "utterance", id: newId("utt"), speaker: who, startS: t, endS: t + text.length / 14, intendedText: who === "bot" ? text : null, spokenText: who === "bot" ? text : null, heardText: text, interrupted: false, ttfbMs: null } });
        t += text.length / 14 + 1; await sleep(500);
      }
      await sleep(800);
      if (stopped) return;
      c.status = "evaluating"; c.durationS = Math.round(t); c.replyP50S = c.timingIsReal ? +(1.1 + Math.random() * 0.8).toFixed(2) : null; c.overlaps = c.timingIsReal ? Math.round(Math.random() * 2) : null;
      run.running--; s.emit({ type: "callDone", call: c });
      await sleep(1500);
      if (stopped) return;
      const r = Math.random();
      c.verdict = r < 0.66 ? "pass" : r < 0.8 ? "fail" : r < 0.94 ? "needs_review" : "invalid";
      c.status = "done"; c.findings = c.verdict === "pass" ? Math.round(Math.random()) : 2 + Math.round(Math.random() * 2); c.openFindings = c.verdict === "pass" ? 0 : c.findings;
      if (c.verdict === "invalid") c.invalidReason = "Caller never gave identifying details before hanging up.";
      run.done++; run.counts[c.verdict === "needs_review" ? "needsReview" : c.verdict]++; run.etaS = Math.max(0, (run.calls - run.done) * 52);
      s.emit({ type: "callEvaluated", call: c }); s.emit({ type: "run", run });
    };
    (async () => {
      const queue = list.filter((c) => c.status === "queued");
      const workers = Array.from({ length: concurrency }, async () => {
        while (queue.length && !stopped) {
          while (run.status === "paused" && !stopped) await sleep(500);
          if (run.status === "stopped") return;
          const c = queue.shift(); if (c) await driveCall(c);
        }
      });
      await Promise.all(workers);
      if (!stopped && run.status === "running") { run.status = "done"; run.endedAt = new Date().toISOString(); run.etaS = null; s.emit({ type: "runDone", run }); }
    })();
  }

  function scriptCall(id: string, s: MockSocket) {
    const summary = summaryOf(id);
    if (!summary) { setTimeout(() => s.close(), 10); return; }
    if (summary.status === "done") { s.emit({ type: "status", status: "done" }); setTimeout(() => s.close(), 50); return; }
    const r = review(id);
    r.eval = null;
    const sc = r.call.scenarioSnapshot;
    const evaluators: T.EvaluatorRun[] = [
      { name: "timing", status: summary.timingIsReal ? "ok" : "skipped", reason: summary.timingIsReal ? "" : "Turn-based call.", durationS: 0.4, model: null, metrics: { reply_p50_s: 1.3, overlaps: 1 } },
      { name: "state", status: summary.mode === "upload" ? "skipped" : "ok", reason: summary.mode === "upload" ? "No database snapshots for an upload." : "", durationS: 0.1, model: null, metrics: {} },
      { name: "pronunciation", status: "ok", reason: "", durationS: 9, model: "whisper-large-v3, no prompt", metrics: {} },
      { name: "transcript_judge", status: sc ? "ok" : "skipped", reason: sc ? "" : "No scenario selected, no rubric.", durationS: 18, model: "Qwen3.6-35B-A3B", metrics: {} },
      { name: "audio_quality", status: "ok", reason: "", durationS: 2, model: "nisqa", metrics: { mos: 4.0 } },
      { name: "audio_listening", status: "ok", reason: "", durationS: 7, model: "LFM3-Audio-2B", metrics: {} },
      { name: "delivery", status: "ok", reason: "", durationS: 2, model: null, metrics: FX.deliveryMetrics(r.speech.bot, []) },
    ];
    let closed = false; s.onCloseInternal = () => { closed = true; };
    (async () => {
      s.emit({ type: "status", status: "evaluating" });
      const findings: T.Finding[] = [];
      for (const [i, e] of evaluators.entries()) {
        await sleep(e.name === "transcript_judge" ? 4000 : 1500);
        if (closed) return;
        s.emit({ type: "evaluator", evaluator: e });
        if (e.name === "pronunciation") {
          const f: T.Finding = { id: `${id}_p1`, evaluator: "pronunciation", category: "critical_token", severity: "major", advisory: false, title: `${sc?.pronunciationWatch[0] ?? "Domain name"} read differently by independent ASR`, detail: "Intended text and ASR disagree on a watch-list token.", evidence: [], clipStartS: r.speech.bot[1]?.[0] ?? 5, clipEndS: r.speech.bot[1]?.[1] ?? 10, criterionId: null, humanVerdict: null, humanNote: "", humanAt: null };
          findings.push(f); s.emit({ type: "finding", finding: f });
        }
        if (e.name === "timing" && e.status === "ok") {
          const f: T.Finding = { id: `${id}_t1`, evaluator: "timing", category: "overlap", severity: "info", advisory: false, title: "1 overlap, bot stopped in 0.3 s", detail: "Within target.", evidence: [], clipStartS: r.speech.caller[2]?.[0] ?? 20, clipEndS: r.speech.caller[2]?.[1] ?? 25, criterionId: null, humanVerdict: null, humanNote: "", humanAt: null };
          findings.push(f); s.emit({ type: "finding", finding: f });
        }
        if (i === evaluators.length - 1) {
          const verdict: T.Verdict = sc && summary.mode !== "upload" ? (findings.some((f) => f.severity === "major") ? "needs_review" : "pass") : "needs_review";
          const ev: T.EvalResult = { callId: id, verdict, summary: sc ? "All rules met; one pronunciation finding to confirm." : "Reduced evidence, review needed.", criteria: sc ? sc.must.map((m) => ({ criterionId: m.id, kind: "must" as const, text: m.text, status: "pass" as const, reason: "", evidence: [] })) : [], findings, evaluators, completedAt: new Date().toISOString() };
          r.eval = ev; summary.status = "done"; summary.verdict = verdict; recount(id);
          s.emit({ type: "eval", eval: ev }); s.emit({ type: "status", status: "done" });
          setTimeout(() => s.close(), 50);
        }
      }
    })();
  }

  return { request, blob, socket, mediaUrl };
}

interface MockSocket extends Socket { emit(data: unknown): void; onCloseInternal: (() => void) | null }
function makeSocket(): MockSocket {
  let open = true;
  const s: MockSocket = {
    onmessage: null, onclose: null, onerror: null, onCloseInternal: null,
    send() { /* mock ignores client messages */ },
    close() { if (!open) return; open = false; s.onCloseInternal?.(); setTimeout(() => s.onclose?.(), 0); },
    emit(data) { if (open) setTimeout(() => s.onmessage?.(data), 0); },
  };
  return s;
}

function renderPrompt(p: T.Persona): string {
  const pat = ["hang up if stalled for more than one turn", "hang up if stalled for more than two turns", "give the agent a fair chance", "wait patiently", "wait as long as it takes"][p.patience - 1];
  const verb = ["answer in a word or two", "keep answers short", "answer normally", "add detail and asides", "ramble and tell stories"][p.verbosity - 1];
  const intr = p.interruptionRate >= 0.75 ? "You interrupt whenever you have something to say, even mid-sentence." : p.interruptionRate >= 0.25 ? "You sometimes cut in before the agent finishes." : "You let the agent finish speaking.";
  return `You are ${p.name}${p.age ? `, ${p.age}` : ""}. You are ${p.temperament}: you ${pat}. You ${verb}. ${p.speakingStyle} ${intr} You stick to the fact sheet and never invent account details.`;
}

function draft(b: { description: string; customerId: string | null }): T.ScenarioDraft {
  const cust = FX.customers.find((c) => c.id === b.customerId) ?? FX.customers[1];
  const scenario: T.ScenarioInput = {
    title: "Website suspended after payment", description: b.description, goal: "Your site shows suspended even though you paid two days ago. You want it back up now and want to know if there is a reinstatement fee.",
    customerId: cust.id, facts: { name: cust.name, domain: cust.domain ?? "", billing_zip: "97035", pin: "7316", invoice_id: "FH-4502", paid_on: "October 28" },
    beats: [
      { id: "state_problem", instruction: "Say the site is down and you already paid.", checkpoint: "The caller said the site is suspended and paid.", required: true },
      { id: "identify", instruction: "Give billing zip and PIN when asked.", checkpoint: "The caller gave identifying details.", required: true },
      { id: "push_back", instruction: "Push back if told to wait 24 hours.", checkpoint: "The caller pushed back on a wait.", required: true },
      { id: "ask_fee", instruction: "Ask whether there is a reinstatement fee.", checkpoint: "The caller asked about a fee.", required: false },
      { id: "confirmation", instruction: "Ask for written confirmation.", checkpoint: "The caller asked for confirmation.", required: false },
    ],
    must: [{ id: "verify_first", text: "Verify the caller before discussing the account.", severity: "critical" }, { id: "confirm_payment", text: "Confirm the payment was received.", severity: "major" }, { id: "reinstates", text: "Reinstate the site with the reinstate_site tool.", severity: "critical" }, { id: "no_fee", text: "State there is no reinstatement fee.", severity: "minor" }, { id: "email", text: "Offer an email confirmation.", severity: "minor" }],
    mustNot: [{ id: "pay_again", text: "Ask the caller to pay again.", severity: "critical" }, { id: "long_wait", text: "Promise a timeframe longer than one hour.", severity: "major" }],
    expectedState: [{ id: "online", table: "websites", op: "changed", where: { domain: cust.domain }, changes: { status: "online" }, description: "Site reinstated." }, { id: "no_fee_row", table: "invoices", op: "no_change", where: { customer_id: cust.id, description: "Reinstatement fee" }, changes: {}, description: "No fee invoice." }],
    maxTurns: 24, pronunciationWatch: ["FH-4502", cust.domain ?? "", "$29.99"], tags: ["billing", "hosting"],
  };
  return { scenario, guessedFields: ["customerId", "facts.paid_on", "facts.invoice_id"], notes: "Mapped to the suspended-site customer in the seed. Check the paid-on date against the payments table.", model: "Qwen3.5-9B" };
}
