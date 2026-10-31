/* Fernhill Internet sample data. Ids and facts follow seeds/ (US English, US dollars). */
import type * as T from "../types";

const now = new Date("2026-10-30T10:04:00-07:00");
export const iso = (minutesAgo: number) => new Date(now.getTime() - minutesAgo * 60_000).toISOString();

const line = (p: Partial<T.LineConditions> = {}): T.LineConditions => ({ codec: "mulaw_8k", noise: "none", snrDb: 20, packetLossPct: 0, oneWayLatencyMs: 80, ...p });
const persona = (p: Partial<T.Persona> & Pick<T.Persona, "id" | "name">): T.Persona => ({
  summary: "", age: null, temperament: "neutral", speakingStyle: "", patience: 3, verbosity: 3, techSavviness: 3, interruptionRate: 0,
  voice: { provider: null, voiceId: "default_female_1", style: null }, line: line(), createdAt: iso(60 * 24 * 9), updatedAt: iso(60 * 24 * 2), ...p,
});

export const personas: T.Persona[] = [
  persona({ id: "per_angry_customer", name: "Angry customer who feels wronged", summary: "Feels cheated by billing and expects an apology before anything else.", age: 53, temperament: "angry", speakingStyle: "Loud and sharp. Repeats the complaint, says \"unbelievable\" and \"this is ridiculous\", and threatens to take their business elsewhere. Never swears.", patience: 1, verbosity: 3, techSavviness: 2, interruptionRate: 0.4, voice: { provider: null, voiceId: "default_male_2", style: null }, line: line({ noise: "street", snrDb: 14 }) }),
  persona({ id: "per_chatty_polite", name: "Chatty, polite retiree", summary: "Friendly, tells stories, thanks the agent a lot.", age: 71, temperament: "friendly", speakingStyle: "Long sentences with asides about grandchildren. Apologizes for taking time.", patience: 5, verbosity: 5, techSavviness: 2, interruptionRate: 0, voice: { provider: null, voiceId: "default_female_2", style: null }, line: line({ noise: "hum", snrDb: 22 }) }),
  persona({ id: "per_confused_retiree", name: "Confused retiree", summary: "Does not know what DNS or IMAP are and mixes up terms.", age: 76, temperament: "anxious", speakingStyle: "Slow, asks the agent to repeat, reads things off a notepad.", patience: 4, verbosity: 3, techSavviness: 1, interruptionRate: 0, voice: { provider: null, voiceId: "default_male_3", style: null }, line: line({ noise: "tv", snrDb: 16 }) }),
  persona({ id: "per_impatient_owner", name: "Impatient small business owner", summary: "Runs a small shop, is losing money while something is broken, and wants it fixed now.", age: 47, temperament: "irritated", speakingStyle: "Clipped sentences. Mentions lost customers and being busy. Sighs and says \"look\" and \"okay, so\" a lot.", patience: 2, verbosity: 2, techSavviness: 2, interruptionRate: 0.25, voice: { provider: null, voiceId: "default_male_1", style: null }, line: line({ noise: "office", snrDb: 18 }) }),
  persona({ id: "per_noisy_cafe_student", name: "Student in a noisy cafe", summary: "Calls from a cafe between classes, distracted.", age: 21, temperament: "neutral", speakingStyle: "Fast, casual, says \"like\" and \"basically\".", patience: 2, verbosity: 3, techSavviness: 4, interruptionRate: 0.1, voice: { provider: null, voiceId: "default_female_3", style: null }, line: line({ noise: "cafe", snrDb: 10, packetLossPct: 2 }) }),
  persona({ id: "per_rushed_interrupter", name: "Rushed interrupter", summary: "Talks over the agent whenever they think they know the answer.", age: 38, temperament: "impatient", speakingStyle: "Starts answering before the question is finished. Short, fast.", patience: 1, verbosity: 2, techSavviness: 3, interruptionRate: 0.75, voice: { provider: null, voiceId: "default_male_4", style: null }, line: line({ noise: "car", snrDb: 12, packetLossPct: 3 }) }),
  persona({ id: "per_skeptical_bookkeeper", name: "Skeptical bookkeeper", summary: "Wants invoice numbers, dates, and amounts read back exactly.", age: 58, temperament: "neutral", speakingStyle: "Precise. Reads references digit by digit and asks for a ticket number.", patience: 3, verbosity: 3, techSavviness: 3, interruptionRate: 0.1, voice: { provider: null, voiceId: "default_female_4", style: null }, line: line({ codec: "none", snrDb: 30 }) }),
  persona({ id: "per_soft_spoken_user", name: "Soft-spoken user", summary: "Quiet voice, hard to hear over the line.", age: 34, temperament: "calm", speakingStyle: "Quiet, short answers, trails off.", patience: 4, verbosity: 1, techSavviness: 3, interruptionRate: 0, voice: { provider: null, voiceId: "default_female_5", style: null }, line: line({ snrDb: 12 }) }),
  persona({ id: "per_sysadmin", name: "Sysadmin who already diagnosed it", summary: "Knows exactly what is wrong and wants the agent to just do it.", age: 41, temperament: "neutral", speakingStyle: "Technical terms, dig output, mild condescension.", patience: 2, verbosity: 3, techSavviness: 5, interruptionRate: 0.3, voice: { provider: null, voiceId: "default_male_5", style: null }, line: line({ codec: "none", snrDb: 32 }) }),
];

const beat = (id: string, instruction: string, checkpoint: string, required = true): T.Beat => ({ id, instruction, checkpoint, required });
const rule = (id: string, text: string, severity: T.Rule["severity"] = "major"): T.Rule => ({ id, text, severity });
const scenario = (s: Partial<T.Scenario> & Pick<T.Scenario, "id" | "title" | "goal">): T.Scenario => ({
  description: "", customerId: null, facts: {}, beats: [], must: [], mustNot: [], expectedState: [], maxTurns: 24, pronunciationWatch: [], tags: [],
  createdAt: iso(60 * 24 * 9), updatedAt: iso(60 * 24 * 1), ...s,
});

export const scenarios: T.Scenario[] = [
  scenario({
    id: "scn_double_charge_refund", title: "Double charge refund", description: "One invoice was charged twice. The bot should refund only the duplicate, which is within its limit.",
    goal: "You were charged twice for your October invoice and you want the extra charge refunded.", customerId: "cus_1001",
    facts: { name: "Dana Whitfield", phone: "(503) 555-0101", domain: "whitfieldlandscaping.example", billing_zip: "97214", pin: "4821", invoice_id: "FH-4419", amount_charged_each_time: "$49.99", charge_date: "October 18" },
    beats: [
      beat("state_problem", "Say you see two charges for the same invoice on your card and you want one back.", "The caller said they were charged twice."),
      beat("identify", "Give your domain or phone so the agent can find your account, then your billing zip and PIN when asked.", "The caller gave identifying details."),
      beat("confirm_refund", "Give the invoice id if asked, and say yes when the agent offers to refund the duplicate.", "The caller agreed to a refund of one charge."),
      beat("close", "Ask when the money comes back to your card, then say thanks and goodbye.", "The caller closed the call.", false),
    ],
    must: [
      rule("verify_first", "Verify the caller with billing zip and PIN before discussing invoice details.", "critical"),
      rule("confirm_amount", "State the $49.99 refund amount and get a yes before issuing it.", "major"),
      rule("refund_timeframe", "State the refund timeframe as 5 to 7 business days.", "major"),
      rule("issues_refund", "Issue the refund with the issue_refund tool.", "critical"),
      rule("offers_ticket", "Offer a ticket or reference number.", "minor"),
    ],
    mustNot: [rule("full_double_refund", "Refund both payments on the invoice.", "critical"), rule("charge_again", "Charge the customer's card again.", "critical")],
    expectedState: [
      { id: "refund_added", table: "refunds", op: "row_added", where: { invoice_id: "FH-4419", amount_cents: 4999 }, changes: {}, description: "One refund of the duplicate payment." },
      { id: "invoice_stays_paid", table: "invoices", op: "no_change", where: { id: "FH-4419" }, changes: {}, description: "The invoice was validly paid once and stays paid." },
    ],
    maxTurns: 20, pronunciationWatch: ["FH-4419", "$49.99", "whitfieldlandscaping.example"], tags: ["billing", "refund", "double-charge"],
  }),
  scenario({
    id: "scn_site_suspended_nonpayment", title: "Website suspended for non-payment", description: "The site is suspended because FH-4502 is unpaid. The bot should take payment and reinstate, or explain what is needed.",
    goal: "Your website is down and shows suspended. You want it back online today.", customerId: "cus_1002",
    facts: { name: "Marcus Bell", phone: "(503) 555-0102", domain: "bellhardware.example", billing_zip: "97035", pin: "7316", invoice_id: "FH-4502", amount: "$29.99" },
    beats: [
      beat("state_problem", "Say the site shows suspended and customers are calling you about it.", "The caller said the site is suspended."),
      beat("identify", "Give your domain, then billing zip and PIN when asked.", "The caller gave identifying details."),
      beat("pay", "When told the invoice is unpaid, say to charge the card on file.", "The caller agreed to pay FH-4502."),
      beat("confirm_online", "Ask how long until the site is back and ask for a confirmation email.", "The caller asked for confirmation.", false),
    ],
    must: [rule("verify_first", "Verify the caller before discussing the account.", "critical"), rule("names_invoice", "Say which invoice is unpaid and the amount.", "major"), rule("reinstates", "Reinstate the site after payment with the reinstate_site tool.", "critical"), rule("states_timeframe", "Say the site is back within 15 minutes.", "minor")],
    mustNot: [rule("waive_without_payment", "Reinstate without taking payment.", "critical"), rule("promise_refund", "Promise a refund of the late fee.", "major")],
    expectedState: [
      { id: "paid", table: "invoices", op: "changed", where: { id: "FH-4502" }, changes: { status: "paid" }, description: "FH-4502 marked paid." },
      { id: "online", table: "websites", op: "changed", where: { domain: "bellhardware.example" }, changes: { status: "online" }, description: "Site reinstated." },
    ],
    maxTurns: 26, pronunciationWatch: ["FH-4502", "$29.99", "bellhardware.example"], tags: ["billing", "hosting"],
  }),
  scenario({
    id: "scn_dns_mx_fix", title: "MX record pointing at the old mail host", description: "Email is not arriving because the MX record points at an old host.", goal: "Your email stopped arriving after you moved hosting. You want it fixed.", customerId: "cus_1007",
    facts: { name: "Walter Nguyen", phone: "(408) 555-0107", domain: "nguyenauto.example", billing_zip: "95112", pin: "5521", current_mx: "mx.oldmailhost.example" },
    beats: [beat("state_problem", "Say customers' emails are bouncing since last week.", "The caller said email is not arriving."), beat("identify", "Give domain, zip, and PIN when asked.", "The caller gave identifying details."), beat("accept_fix", "Say yes when the agent offers to update the MX record.", "The caller agreed to the DNS change.")],
    must: [rule("verify_first", "Verify the caller before changing DNS.", "critical"), rule("explains_mx", "Explain that the MX record points at mx.oldmailhost.example.", "major"), rule("updates_mx", "Update the MX record to mail.fernhill.example with the update_dns_record tool.", "critical"), rule("ttl_warning", "Say the change can take up to an hour to propagate.", "minor")],
    mustNot: [rule("delete_a_record", "Delete or change the A record.", "critical")],
    expectedState: [{ id: "mx_fixed", table: "dns_records", op: "changed", where: { domain: "nguyenauto.example", type: "MX" }, changes: { value: "mail.fernhill.example" }, description: "MX updated." }],
    maxTurns: 24, pronunciationWatch: ["MX", "mail.fernhill.example", "nguyenauto.example"], tags: ["dns", "email"],
  }),
  scenario({ id: "scn_already_refunded", title: "Refund of an already refunded invoice", goal: "You want a refund for your September invoice, and you don't believe it was already done.", customerId: "cus_1012", facts: { name: "Linda Kowalski", invoice_id: "FH-4288", amount: "$19.99" }, maxTurns: 22, tags: ["billing", "refund", "refuse"], pronunciationWatch: ["FH-4288", "$19.99"], must: [rule("verify_first", "Verify the caller before discussing the invoice.", "critical"), rule("tells_truth", "Say that FH-4288 was already refunded on September 12.", "major"), rule("offers_ticket", "Offer a ticket so a person can trace the earlier refund.", "major")], mustNot: [rule("refund_again", "Issue another refund on FH-4288.", "critical")] }),
  scenario({ id: "scn_domain_renewal", title: "Domain about to expire", goal: "Your domain expires next week and you want to renew it and turn auto-renew on.", customerId: "cus_1009", facts: { name: "Greg Thompson", domain: "thompsonphoto.example", expires: "November 6", price: "$14.99" }, tags: ["domains", "billing"], pronunciationWatch: ["thompsonphoto.example", "$14.99"] }),
  scenario({ id: "scn_expired_ssl", title: "Expired SSL certificate", goal: "Your site shows a security warning and you want it fixed.", customerId: "cus_1008", facts: { name: "Sofia Marino", domain: "marinocafe.example", expired_on: "October 21" }, tags: ["ssl", "hosting"], pronunciationWatch: ["SSL", "marinocafe.example"] }),
  scenario({ id: "scn_failed_verification", title: "Caller cannot verify identity", goal: "You want to change the card on file but you do not remember your PIN.", customerId: "cus_1006", facts: { name: "Renee Castillo", domain: "castillostudio.example" }, tags: ["security"] }),
  scenario({ id: "scn_forgotten_password", title: "Forgotten mailbox password", goal: "You cannot log in to your email and want the password reset.", customerId: "cus_1004", facts: { name: "Priya Nair", mailbox: "priya@nairdesign.example" }, tags: ["email"], pronunciationWatch: ["nairdesign.example"] }),
  scenario({ id: "scn_locked_mailbox", title: "Locked mailbox after failed logins", goal: "Your email says the account is locked.", customerId: "cus_1003", facts: { name: "Evelyn Hart", mailbox: "evelyn@hartfamily.example" }, tags: ["email"] }),
  scenario({ id: "scn_past_due_expired_card", title: "Past due with an expired card", goal: "You got a past due notice and need to update your card.", customerId: "cus_1011", facts: { name: "Paul Jensen", invoice_id: "FH-4588", amount: "$19.99" }, tags: ["billing"], pronunciationWatch: ["FH-4588"] }),
  scenario({ id: "scn_plan_upgrade_full_mailbox", title: "Mailbox full, wants an upgrade", goal: "Your mailbox is full and you want more space.", customerId: "cus_1010", facts: { name: "Aisha Rahman", mailbox: "aisha@rahmanlaw.example", plan: "Email Basic" }, tags: ["email", "billing"] }),
  scenario({ id: "scn_refund_over_limit", title: "Refund above the phone limit", goal: "You want the $189 annual prepay refunded.", customerId: "cus_1005", facts: { name: "Tom Okafor", invoice_id: "FH-4631", amount: "$189.00" }, tags: ["billing", "refund"], pronunciationWatch: ["FH-4631", "$189.00"] }),
];

export const customers: T.Customer[] = [
  { id: "cus_1001", name: "Dana Whitfield", note: "Charged twice for invoice FH-4419.", plans: ["Website Pro"], domain: "whitfieldlandscaping.example" },
  { id: "cus_1002", name: "Marcus Bell", note: "Website suspended for non-payment of FH-4502.", plans: ["Website Business"], domain: "bellhardware.example" },
  { id: "cus_1003", name: "Evelyn Hart", note: "Mailbox evelyn@hartfamily.example is locked after failed logins.", plans: ["Email Basic"], domain: "hartfamily.example" },
  { id: "cus_1004", name: "Priya Nair", note: "Forgot the password for priya@nairdesign.example.", plans: ["Email Plus"], domain: "nairdesign.example" },
  { id: "cus_1005", name: "Tom Okafor", note: "Paid $189.00 for a year on FH-4631, above the phone refund limit.", plans: ["Website Business"], domain: "okaforconsulting.example" },
  { id: "cus_1006", name: "Renee Castillo", note: "Ordinary account, used for failed identity verification tests.", plans: ["Website Starter"], domain: "castillostudio.example" },
  { id: "cus_1007", name: "Walter Nguyen", note: "MX record points at an old mail host, so email is not arriving.", plans: ["Email Plus"], domain: "nguyenauto.example" },
  { id: "cus_1008", name: "Sofia Marino", note: "SSL certificate on marinocafe.example expired on 2026-10-21.", plans: ["Website Starter"], domain: "marinocafe.example" },
  { id: "cus_1009", name: "Greg Thompson", note: "Domain thompsonphoto.example expires on 2026-11-06 with auto renew off.", plans: ["Website Starter"], domain: "thompsonphoto.example" },
  { id: "cus_1010", name: "Aisha Rahman", note: "Mailbox aisha@rahmanlaw.example is full on the Email Basic plan.", plans: ["Email Basic"], domain: "rahmanlaw.example" },
  { id: "cus_1011", name: "Paul Jensen", note: "Past due on FH-4588, and the card on file has expired.", plans: ["Website Starter"], domain: "jensenplumbing.example" },
  { id: "cus_1012", name: "Linda Kowalski", note: "Two refunds already issued this year, on FH-4205 and FH-4288.", plans: ["Website Starter"], domain: "kowalskicrafts.example" },
];

export const profiles: T.Profile[] = [
  { id: "local-default", label: "local-default", description: "Qwen3.5-9B bot, Qwen3.5-4B caller, whisper STT, Qwen3-TTS", configPath: "config/local-default.yaml", health: "ok", liveCapable: true, benchmark: { botReplyP50S: 1.42, at: iso(134) }, untested: false },
  { id: "local-35b", label: "local-qwen36-35b", description: "Qwen3.6-35B-A3B bot, same caller and audio stack", configPath: "config/local-35b.yaml", health: "ok", liveCapable: false, benchmark: { botReplyP50S: 3.1, at: iso(900) }, untested: false },
  { id: "local-lfm3", label: "local-lfm3-realtime", description: "LFM3-Audio-2B speech-to-speech bot", configPath: "config/local-lfm3.yaml", health: "loading", liveCapable: true, benchmark: { botReplyP50S: 0.74, at: iso(1300) }, untested: false },
  { id: "bedrock-nova", label: "bedrock-nova", description: "Nova Sonic bot, Claude judge. Adapters not tested against live AWS.", configPath: "config/bedrock-nova.yaml", health: "down", liveCapable: true, benchmark: null, untested: true },
];

export const health: T.Health[] = [
  { role: "llm_support", name: "support bot LLM", model: "Qwen3.5-9B-Instruct (llama.cpp :8101)", state: "ok", detail: "ready, 38 tok/s", elapsedS: null, usualS: null },
  { role: "llm_caller", name: "caller LLM", model: "Qwen3.5-4B (llama.cpp :8102)", state: "ok", detail: "ready", elapsedS: null, usualS: null },
  { role: "stt_live", name: "STT", model: "faster-whisper large-v3 (:8110)", state: "ok", detail: "ready, RTF 0.21", elapsedS: null, usualS: null },
  { role: "tts_bot", name: "TTS", model: "Qwen3-TTS 1.7B CustomVoice (:8120)", state: "ok", detail: "ready, first byte 410 ms", elapsedS: null, usualS: null },
  { role: "llm_judge", name: "judge LLM", model: "Qwen3.6-35B-A3B (llama.cpp :8103)", state: "loading", detail: "loading weights", elapsedS: 22, usualS: 40 },
  { role: "audio_analyst", name: "audio analyst", model: "LFM3-Audio-2B (:8130)", state: "ok", detail: "ready", elapsedS: null, usualS: null },
  { role: "mos", name: "MOS predictor", model: "NISQA (in-process)", state: "ok", detail: "ready", elapsedS: null, usualS: null },
];

const counts = (pass: number, fail: number, needsReview: number, invalid: number): T.Counts => ({ pass, fail, needsReview, invalid });
export const runs: T.RunSummary[] = [
  { id: "batch_0142", label: "all 12 scenarios x 4 personas", profile: "local-default", mode: "live_bots", status: "done", startedAt: iso(47), endedAt: iso(23), calls: 48, done: 48, running: 0, counts: counts(31, 9, 6, 2), etaS: null, scenarioIds: scenarios.map((s) => s.id), personaIds: ["per_impatient_owner", "per_chatty_polite", "per_skeptical_bookkeeper", "per_rushed_interrupter"], reps: 1 },
  { id: "batch_0141", label: "refund and suspension x 4 personas x 3", profile: "local-35b", mode: "turn_based", status: "done", startedAt: iso(122), endedAt: iso(68), calls: 24, done: 24, running: 0, counts: counts(19, 2, 3, 0), etaS: null, scenarioIds: ["scn_double_charge_refund", "scn_site_suspended_nonpayment"], personaIds: ["per_impatient_owner", "per_chatty_polite", "per_skeptical_bookkeeper", "per_rushed_interrupter"], reps: 3 },
  { id: "batch_0140", label: "refund and suspension x 4 personas x 3", profile: "local-default", mode: "live_bots", status: "done", startedAt: iso(60 * 16 + 24), endedAt: iso(60 * 15 + 40), calls: 24, done: 24, running: 0, counts: counts(17, 4, 2, 1), etaS: null, scenarioIds: ["scn_double_charge_refund", "scn_site_suspended_nonpayment"], personaIds: ["per_impatient_owner", "per_chatty_polite", "per_skeptical_bookkeeper", "per_rushed_interrupter"], reps: 3 },
  { id: "batch_0139", label: "mic calls, dan", profile: "local-default", mode: "human", status: "done", startedAt: iso(60 * 18), endedAt: iso(60 * 17 + 40), calls: 3, done: 3, running: 0, counts: counts(2, 0, 1, 0), etaS: null, scenarioIds: ["scn_site_suspended_nonpayment"], personaIds: [], reps: 1 },
  { id: "batch_0138", label: "forgotten password x 4 personas x 3", profile: "local-lfm3", mode: "live_bots", status: "done", startedAt: iso(60 * 20), endedAt: iso(60 * 19 + 20), calls: 12, done: 12, running: 0, counts: counts(4, 6, 1, 1), etaS: null, scenarioIds: ["scn_forgotten_password"], personaIds: ["per_confused_retiree", "per_chatty_polite", "per_noisy_cafe_student", "per_soft_spoken_user"], reps: 3 },
  { id: "batch_0137", label: "uploaded recording", profile: "local-default", mode: "upload", status: "done", startedAt: iso(60 * 22), endedAt: iso(60 * 22 - 2), calls: 1, done: 1, running: 0, counts: counts(0, 0, 1, 0), etaS: null, scenarioIds: [], personaIds: [], reps: 1 },
];

const pname = (id: string | null) => personas.find((p) => p.id === id)?.name ?? null;
const stitle = (id: string | null) => scenarios.find((s) => s.id === id)?.title ?? null;
const call = (c: Partial<T.CallSummary> & Pick<T.CallSummary, "id" | "mode" | "verdict">): T.CallSummary => ({
  batchId: null, rep: 1, scenarioId: null, personaId: null, status: "done", profile: "local-default", humanVerdict: null, invalidReason: null,
  findings: 0, openFindings: 0, durationS: 180, replyP50S: 1.4, overlaps: 0, startedAt: iso(40), timingIsReal: c.mode !== "turn_based",
  ...c, scenarioTitle: stitle(c.scenarioId ?? null), personaName: c.mode === "human" ? "you" : pname(c.personaId ?? null),
});

export const calls: T.CallSummary[] = [
  call({ id: "call_a3f2c1d0e9b7", batchId: "batch_0142", scenarioId: "scn_double_charge_refund", personaId: "per_impatient_owner", mode: "live_bots", verdict: "needs_review", findings: 6, openFindings: 4, durationS: 172, replyP50S: 1.42, overlaps: 1, startedAt: iso(43) }),
  call({ id: "call_b801d2e3f4a5", batchId: "batch_0142", scenarioId: "scn_double_charge_refund", personaId: "per_chatty_polite", mode: "live_bots", verdict: "pass", findings: 1, openFindings: 0, durationS: 241, replyP50S: 1.31, startedAt: iso(42) }),
  call({ id: "call_c44e5f6a7b8c", batchId: "batch_0142", scenarioId: "scn_dns_mx_fix", personaId: "per_skeptical_bookkeeper", mode: "live_bots", verdict: "fail", findings: 3, openFindings: 3, durationS: 198, replyP50S: 1.55, overlaps: 1, startedAt: iso(41) }),
  call({ id: "call_d1b96c7d8e9f", batchId: "batch_0142", scenarioId: "scn_site_suspended_nonpayment", personaId: "per_rushed_interrupter", mode: "live_bots", verdict: "needs_review", findings: 4, openFindings: 4, durationS: 263, replyP50S: 1.38, overlaps: 3, startedAt: iso(39) }),
  call({ id: "call_e07c8d9e0f1a", batchId: "batch_0142", scenarioId: "scn_already_refunded", personaId: "per_impatient_owner", mode: "live_bots", verdict: "invalid", findings: 1, openFindings: 0, durationS: 61, replyP50S: 1.47, startedAt: iso(37), invalidReason: "Caller gave invoice FH-4228 instead of FH-4288 and hung up before the scenario started." }),
  call({ id: "call_f2aa9e0f1a2b", batchId: "batch_0142", scenarioId: "scn_forgotten_password", personaId: "per_chatty_polite", mode: "live_bots", verdict: "pass", findings: 0, openFindings: 0, durationS: 227, replyP50S: 1.29, startedAt: iso(35) }),
  call({ id: "call_1c3d0a1b2c3d", batchId: "batch_0141", scenarioId: "scn_double_charge_refund", personaId: "per_impatient_owner", mode: "turn_based", profile: "local-35b", verdict: "needs_review", findings: 3, openFindings: 2, durationS: 164, replyP50S: null, overlaps: null, rep: 2, startedAt: iso(110) }),
  call({ id: "call_2e901b2c3d4e", batchId: "batch_0141", scenarioId: "scn_site_suspended_nonpayment", personaId: "per_chatty_polite", mode: "turn_based", profile: "local-35b", verdict: "pass", findings: 1, openFindings: 0, durationS: 255, replyP50S: null, overlaps: null, startedAt: iso(108) }),
  call({ id: "call_3f172c3d4e5f", batchId: "batch_0141", scenarioId: "scn_double_charge_refund", personaId: "per_skeptical_bookkeeper", mode: "turn_based", profile: "local-35b", verdict: "fail", findings: 2, openFindings: 2, durationS: 188, replyP50S: null, overlaps: null, rep: 3, startedAt: iso(100) }),
  call({ id: "call_h0013d4e5f6a", batchId: "batch_0139", scenarioId: "scn_site_suspended_nonpayment", personaId: null, mode: "human", verdict: "needs_review", findings: 2, openFindings: 2, durationS: 143, replyP50S: 1.21, overlaps: 4, startedAt: iso(60 * 18) }),
  call({ id: "call_9a2b4e5f6a7b", batchId: "batch_0138", scenarioId: "scn_forgotten_password", personaId: "per_confused_retiree", mode: "live_bots", profile: "local-lfm3", verdict: "fail", findings: 5, openFindings: 1, durationS: 302, replyP50S: 0.74, overlaps: 6, startedAt: iso(60 * 20) }),
  call({ id: "call_u0015f6a7b8c", batchId: "batch_0137", scenarioId: null, personaId: null, mode: "upload", verdict: "needs_review", findings: 2, openFindings: 2, durationS: 311, replyP50S: 1.9, overlaps: 1, startedAt: iso(60 * 22) }),
];

export const queue: T.QueueItem[] = [
  { callId: "call_u0015f6a7b8c", scenarioId: null, scenarioTitle: null, personaName: "upload", openFindings: 2, why: "reduced evidence, judge only", createdAt: iso(60 * 22) },
  { callId: "call_h0013d4e5f6a", scenarioId: "scn_site_suspended_nonpayment", scenarioTitle: "Website suspended for non-payment", personaName: "you (mic)", openFindings: 2, why: "4 overlaps, pronunciation of bellhardware", createdAt: iso(60 * 18) },
  { callId: "call_1c3d0a1b2c3d", scenarioId: "scn_double_charge_refund", scenarioTitle: "Double charge refund", personaName: "Impatient small business owner", openFindings: 2, why: "zip misread, vague timeframe", createdAt: iso(110) },
  { callId: "call_3f172c3d4e5f", scenarioId: "scn_double_charge_refund", scenarioTitle: "Double charge refund", personaName: "Skeptical bookkeeper", openFindings: 2, why: "fail, confirm before counting", createdAt: iso(100) },
  { callId: "call_a3f2c1d0e9b7", scenarioId: "scn_double_charge_refund", scenarioTitle: "Double charge refund", personaName: "Impatient small business owner", openFindings: 4, why: "wrong refund timeframe, no refund issued", createdAt: iso(43) },
  { callId: "call_c44e5f6a7b8c", scenarioId: "scn_dns_mx_fix", scenarioTitle: "MX record pointing at the old mail host", personaName: "Skeptical bookkeeper", openFindings: 3, why: "MX value read wrong twice", createdAt: iso(41) },
  { callId: "call_d1b96c7d8e9f", scenarioId: "scn_site_suspended_nonpayment", scenarioTitle: "Website suspended for non-payment", personaName: "Rushed interrupter", openFindings: 4, why: "3 overlaps, judge uncertain on names_invoice", createdAt: iso(39) },
];

export const scenarioRates = [
  { scenarioId: "scn_dns_mx_fix", counts: counts(18, 2, 1, 1) },
  { scenarioId: "scn_forgotten_password", counts: counts(20, 6, 3, 1) },
  { scenarioId: "scn_double_charge_refund", counts: counts(21, 8, 4, 1) },
  { scenarioId: "scn_site_suspended_nonpayment", counts: counts(27, 3, 3, 1) },
  { scenarioId: "scn_expired_ssl", counts: counts(11, 0, 1, 0) },
  { scenarioId: "scn_already_refunded", counts: counts(9, 4, 2, 1) },
  { scenarioId: "scn_domain_renewal", counts: counts(7, 1, 0, 0) },
  { scenarioId: "scn_locked_mailbox", counts: counts(6, 2, 1, 0) },
].map((r) => ({ ...r, scenarioTitle: stitle(r.scenarioId) ?? r.scenarioId }));

export const topOpenFindings = [
  { title: "Invoice or reference number mispronounced", evaluator: "pronunciation", count: 14 },
  { title: "Refund timeframe wrong or vague", evaluator: "transcript_judge", count: 11 },
  { title: "Bot continued more than 0.5 s after barge-in", evaluator: "timing", count: 9 },
  { title: "MX record value read wrong", evaluator: "pronunciation", count: 7 },
  { title: "Expected database change missing", evaluator: "state", count: 6 },
  { title: "Flat delivery on apology", evaluator: "audio_listening", count: 5 },
];

export const matrix: T.MatrixCell[] = (() => {
  const base: Record<string, number[]> = {
    scn_double_charge_refund: [50, 72, 68, 58, 80, 45, 70, 85, 66], scn_site_suspended_nonpayment: [72, 88, 90, 75, 85, 60, 82, 92, 80],
    scn_dns_mx_fix: [78, 92, 85, 80, 88, 70, 90, 95, 97], scn_forgotten_password: [60, 70, 75, 55, 65, 40, 72, 80, 85],
    scn_already_refunded: [45, 70, 66, 50, 60, 30, 75, 80, 70], scn_expired_ssl: [90, 100, 92, 85, 90, 80, 95, 100, 100],
  };
  const out: T.MatrixCell[] = [];
  for (const s of scenarios) personas.forEach((p, i) => {
    const pct = base[s.id]?.[i];
    if (pct == null) return;
    const n = 6;
    const pass = Math.round((pct / 100) * n);
    out.push({ scenarioId: s.id, personaId: p.id, counts: counts(pass, n - pass, 0, 0) });
  });
  return out;
})();

export const voices: T.Voice[] = [
  ...["default_male_1", "default_male_2", "default_male_3", "default_male_4", "default_male_5", "default_female_1", "default_female_2", "default_female_3", "default_female_4", "default_female_5"].map((v) => ({ provider: "qwen3_tts", voiceId: v, label: v.replace(/_/g, " ").replace(/^default /, "Qwen3-TTS "), language: "en-US" })),
  { provider: "pocket_tts", voiceId: "default", label: "pocket-tts default", language: "en-US" },
];

export const roles: T.RoleItem[] = [
  { role: "llm_support", label: "support bot LLM", interface: "LLM", adapter: "llamacpp:qwen3.5-9b", options: [{ id: "llamacpp:qwen3.5-9b", label: "Qwen3.5-9B-Instruct · llama.cpp :8101", untested: false, license: "Apache-2.0" }, { id: "llamacpp:qwen3.6-35b", label: "Qwen3.6-35B-A3B · llama.cpp :8103", untested: false, license: "Apache-2.0" }, { id: "llamacpp:gpt-oss-20b", label: "gpt-oss-20b · llama.cpp", untested: false, license: "Apache-2.0" }, { id: "bedrock:claude-sonnet", label: "Bedrock · Claude Sonnet", untested: true, license: "Bedrock terms" }], health: health[0] },
  { role: "realtime", label: "support bot realtime voice", interface: "RealtimeVoice", adapter: "none", options: [{ id: "none", label: "none (STT + LLM + TTS)", untested: false, license: "" }, { id: "lfm3:audio-2b", label: "LFM3-Audio-2B · :8130", untested: false, license: "LFM licence, conditional" }, { id: "bedrock:nova-sonic", label: "Bedrock · Nova Sonic", untested: true, license: "Bedrock terms" }], health: { role: "realtime", name: "support bot realtime voice", model: "", state: "off", detail: "off", elapsedS: null, usualS: null } },
  { role: "llm_caller", label: "caller LLM", interface: "LLM", adapter: "llamacpp:qwen3.5-4b", options: [{ id: "llamacpp:qwen3.5-4b", label: "Qwen3.5-4B · llama.cpp :8102", untested: false, license: "Apache-2.0" }, { id: "llamacpp:qwen3.5-9b", label: "Qwen3.5-9B-Instruct · llama.cpp :8101", untested: false, license: "Apache-2.0" }], health: health[1] },
  { role: "stt_live", label: "STT, live", interface: "STT", adapter: "whisper:large-v3", options: [{ id: "whisper:large-v3", label: "faster-whisper large-v3 · :8110", untested: false, license: "MIT" }, { id: "bedrock:transcribe", label: "Bedrock · Transcribe streaming", untested: true, license: "AWS terms" }], health: health[2] },
  { role: "tts_bot", label: "TTS, bot voice", interface: "TTS", adapter: "qwen3_tts:1.7b", options: [{ id: "qwen3_tts:1.7b", label: "Qwen3-TTS 1.7B CustomVoice · :8120", untested: false, license: "Apache-2.0" }, { id: "pocket_tts", label: "kyutai pocket-tts · :8121", untested: false, license: "CC-BY 4.0" }, { id: "bedrock:polly", label: "Bedrock · Polly", untested: true, license: "AWS terms" }], health: health[3] },
  { role: "llm_judge", label: "judge LLM", interface: "LLM", adapter: "llamacpp:qwen3.6-35b", options: [{ id: "llamacpp:qwen3.6-35b", label: "Qwen3.6-35B-A3B · llama.cpp :8103", untested: false, license: "Apache-2.0" }, { id: "vllm:qwen3-32b-awq", label: "Qwen3-32B-AWQ · vLLM", untested: false, license: "Apache-2.0" }, { id: "bedrock:claude-sonnet", label: "Bedrock · Claude Sonnet", untested: true, license: "Bedrock terms" }], health: health[4] },
  { role: "pronunciation_asr", label: "pronunciation ASR", interface: "STT", adapter: "whisper:large-v3-noprompt", options: [{ id: "whisper:large-v3-noprompt", label: "faster-whisper large-v3, no prompt · :8110", untested: false, license: "MIT" }], health: { ...health[2], role: "pronunciation_asr" } },
  { role: "audio_analyst", label: "audio analyst", interface: "AudioAnalyst", adapter: "lfm3:audio-2b", options: [{ id: "lfm3:audio-2b", label: "LFM3-Audio-2B · :8130", untested: false, license: "LFM licence, conditional" }, { id: "bedrock:nova", label: "Bedrock · Nova", untested: true, license: "Bedrock terms" }], health: health[5] },
  { role: "mos", label: "MOS predictor", interface: "AudioQuality", adapter: "nisqa", options: [{ id: "nisqa", label: "NISQA, in-process", untested: false, license: "MIT" }], health: health[6] },
];

export const system: T.SystemInfo = {
  gpu: { device: "Radeon 8060S, gfx1151", memoryUsedGb: 41.2, memoryTotalGb: 122 },
  servers: [{ role: "llm_support", kind: "llama.cpp", port: 8101, pid: 41211 }, { role: "llm_caller", kind: "llama.cpp", port: 8102, pid: 41240 }, { role: "llm_judge", kind: "llama.cpp", port: 8103, pid: 41377 }, { role: "stt_live", kind: "faster-whisper", port: 8110, pid: 41402 }, { role: "tts_bot", kind: "qwen3-tts", port: 8120, pid: 41455 }, { role: "audio_analyst", kind: "lfm3", port: 8130, pid: 41490 }],
  urls: { local: "https://localhost:8443", lan: "https://192.168.0.75:8443" }, cert: "self-signed",
};

export const thresholds: T.Thresholds = { deadAirS: 2.0, bargeStopTargetS: 0.5, replyLatencyWarnS: 2.5, mosBaseline: 4.0, audioListeningMode: "advisory" };

/* ---------- full review payloads ---------- */

type Seg = [number, number];
type TurnSpec = { who: T.Speaker; t: number; end: number; heard: string; intended?: string; interrupted?: boolean };
type EvSpec = { kind: "tool" | "db" | "barge" | "system"; t: number; name?: string; args?: Record<string, unknown>; result?: unknown; detail?: string; stoppedAt?: number; error?: string; guard?: T.GuardObjection };

function timeline(turns: TurnSpec[], events: EvSpec[], callId: string): T.TimelineEvent[] {
  const out: T.TimelineEvent[] = [];
  turns.forEach((u, i) => out.push({ type: "utterance", id: `utt_${callId.slice(5, 9)}_${i}`, speaker: u.who, startS: u.t, endS: u.end, intendedText: u.who === "bot" ? (u.intended ?? u.heard) : null, spokenText: u.who === "bot" ? u.heard : null, heardText: u.heard, interrupted: !!u.interrupted, ttfbMs: null }));
  events.forEach((e, i) => {
    if (e.kind === "tool") out.push({ type: "tool_call", id: `tc_${i}`, atS: e.t, name: e.name!, arguments: e.args ?? {}, result: e.result ?? null, error: e.error ?? null, durationMs: 120, guard: e.guard ?? null });
    else if (e.kind === "barge") out.push({ type: "barge_in", id: `bi_${i}`, atS: e.t, botStoppedAtS: e.stoppedAt ?? null, interruptedUtteranceId: null });
    else out.push({ type: "system", id: `sys_${i}`, atS: e.t, kind: e.kind === "db" ? "db_change" : "note", detail: e.detail ?? "" });
  });
  return out.sort((a, b) => ("startS" in a ? a.startS : a.atS) - ("startS" in b ? b.startS : b.atS));
}

export function peaksFromSpeech(segs: Seg[], durationS: number, perSecond = 50): number[] {
  const n = Math.ceil(durationS * perSecond);
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const t = i / perSecond;
    let amp = 0.02 + 0.01 * Math.abs(Math.sin(i * 0.7));
    for (const [a, b] of segs) {
      if (t >= a && t <= b) {
        const rel = (t - a) / (b - a);
        const env = Math.min(1, rel * 8, (1 - rel) * 6);
        const r = Math.abs(Math.sin(i * 12.9898 + a * 78.233) * 43758.5453) % 1;
        amp = (0.25 + 0.65 * r) * env * (0.6 + 0.4 * Math.abs(Math.sin(t * 7)));
        break;
      }
    }
    out[i] = Math.min(1, amp);
  }
  return out;
}

const F = (id: string, evaluator: string, category: string, severity: T.Severity, title: string, detail: string, clip: Seg, extra: Partial<T.Finding> = {}): T.Finding => ({
  id, evaluator, category, severity, advisory: false, title, detail, evidence: [], clipStartS: clip[0], clipEndS: clip[1], criterionId: null, humanVerdict: null, humanNote: "", humanAt: null, ...extra,
});

export const reviewA = ((): T.ReviewPayload => {
  const id = "call_a3f2c1d0e9b7";
  const sc = scenarios[0];
  const pe = personas.find((p) => p.id === "per_impatient_owner")!;
  const speech: { caller: Seg[]; bot: Seg[] } = {
    caller: [[1.2, 6.8], [15.4, 21.0], [27.9, 31.2], [38.6, 44.1], [52.0, 54.9], [64.2, 70.5], [84.0, 89.2], [104.5, 109.0], [120.3, 124.1], [138.0, 141.2], [155.6, 160.0], [166.8, 169.4]],
    bot: [[0.0, 2.4], [9.6, 14.2], [22.4, 27.1], [32.5, 37.6], [45.3, 51.1], [56.3, 63.0], [66.9, 69.6], [72.0, 82.6], [90.9, 103.2], [111.0, 119.0], [125.6, 136.4], [142.9, 154.3], [161.2, 165.9], [170.1, 172.0]],
  };
  const turns: TurnSpec[] = [
    { who: "bot", t: 0.0, end: 2.4, heard: "Thanks for calling Fernhill Internet, this is Ada. How can I help?" },
    { who: "caller", t: 1.2, end: 6.8, heard: "Okay, so, I got charged twice this month. Same invoice. I need that fixed today." },
    { who: "bot", t: 9.6, end: 14.2, heard: "Sorry about that. Can I get the billing zip code and PIN on the account?" },
    { who: "caller", t: 15.4, end: 21.0, heard: "Nine seven two one four. PIN is four eight two one. Whitfield Landscaping." },
    { who: "bot", t: 22.4, end: 27.1, heard: "Thanks, Dana. Let me pull up your recent invoices." },
    { who: "caller", t: 27.9, end: 31.2, heard: "It's the one from the eighteenth." },
    { who: "bot", t: 32.5, end: 37.6, heard: "Bear with me one second while that loads." },
    { who: "caller", t: 38.6, end: 44.1, heard: "Forty-nine ninety-nine, twice. Same day. Look, I'm busy here." },
    { who: "bot", t: 45.3, end: 51.1, heard: "I can see invoice FH forty-four ninety was charged twice on October 18th, $49.99 each.", intended: "I can see invoice FH-4419 was charged twice on October 18th, $49.99 each." },
    { who: "caller", t: 52.0, end: 54.9, heard: "Forty-four ninety? No. Forty-four nineteen." },
    { who: "bot", t: 56.3, end: 63.0, heard: "Apologies, FH-4419. That's the one. I'll get a refund raised for the second $49.99 payment." },
    { who: "caller", t: 64.2, end: 70.5, heard: "Fine, and how long is that going to take, because last time it took weeks." },
    { who: "bot", t: 66.9, end: 69.6, heard: "The refund goes back to the", intended: "The refund goes back to the card you paid with.", interrupted: true },
    { who: "bot", t: 72.0, end: 82.6, heard: "I understand. It goes back to the same card you paid with, and I'll add a note so you're not charged twice again next month." },
    { who: "caller", t: 84.0, end: 89.2, heard: "You didn't answer. How long." },
    { who: "bot", t: 90.9, end: 103.2, heard: "You'll see that back on your card within 24 hours, and I'm sorry again for the hassle." },
    { who: "caller", t: 104.5, end: 109.0, heard: "Fine. Put that in writing on the account." },
    { who: "bot", t: 111.0, end: 119.0, heard: "Done, I've added a note to the account with today's date." },
    { who: "caller", t: 120.3, end: 124.1, heard: "And the refund's actually going through, right?" },
    { who: "bot", t: 125.6, end: 136.4, heard: "Yes, I'm very sorry for the inconvenience this has caused, it is being processed now." },
    { who: "caller", t: 138.0, end: 141.2, heard: "Okay. Reference number?" },
    { who: "bot", t: 142.9, end: 154.3, heard: "The refund is in progress and you'll get an email confirmation shortly. Is there anything else I can help with today?" },
    { who: "caller", t: 155.6, end: 160.0, heard: "A reference. Number. For the refund." },
    { who: "bot", t: 161.2, end: 165.9, heard: "Of course. Your reference is FH-R-2291." },
    { who: "caller", t: 166.8, end: 169.4, heard: "Yeah. Bye." },
    { who: "bot", t: 170.1, end: 172.0, heard: "Thanks for calling Fernhill." },
  ];
  const events: EvSpec[] = [
    { kind: "tool", t: 7.9, name: "lookup_customer", args: { phone: "(503) 555-0101" }, result: { id: "cus_1001", name: "Dana Whitfield", status: "active" } },
    { kind: "tool", t: 15.2, name: "verify_customer", args: { customerId: "cus_1001", billingZip: "90210", pin: "1234" }, result: { error: "The caller hasn't said that. Ask them for it." }, error: "blocked before reaching the tool by argument_guard rule not_heard: billingZip, pin not in anything the bot heard from the caller", guard: { rule: "not_heard", fields: ["billingZip", "pin"], stopped: true } },
    { kind: "tool", t: 21.6, name: "verify_customer", args: { customerId: "cus_1001", billingZip: "97214", pin: "4821" }, result: { verified: true } },
    { kind: "tool", t: 31.6, name: "list_invoices", args: { customerId: "cus_1001", months: 3 }, result: [{ id: "FH-4310", amount: 49.99, status: "paid" }, { id: "FH-4419", amount: 49.99, status: "paid", payments: 2 }] },
    { kind: "barge", t: 68.1, stoppedAt: 69.6 },
    { kind: "db", t: 110.2, detail: "INSERT tickets (customer_id=cus_1001, subject=\"Duplicate charge FH-4419, refund requested\")" },
  ];
  const findings: T.Finding[] = [
    F("fnd_a1", "timing", "dead_air", "minor", "Dead air 2.9 s before bot answered", "Caller finished at 00:21.0, bot started at 00:22.4 after a 1.4 s gap, then paused again. Threshold 2.0 s.", [15.4, 22.4]),
    F("fnd_a2", "pronunciation", "critical_token_corroborated", "major", "Invoice number read as \"FH forty-four ninety\"", "Intended: \"invoice FH-4419\". Independent ASR heard: \"invoice FH forty-four ninety\"; the bot's own STT heard the same. Critical token mismatch on a reference number.", [45.3, 51.1], { criterionId: "FH-4419", evidence: [{ quote: "I can see invoice FH forty-four ninety was charged twice", speaker: "bot", startS: 45.3, endS: 51.1 }] }),
    F("fnd_a3", "timing", "barge_in_stop", "info", "Bot kept talking 1.5 s after barge-in", "Caller interrupted at 01:08.1. Bot audio continued until 01:09.6. Target under 0.5 s.", [66.9, 70.5]),
    F("fnd_a4", "transcript_judge", "rule", "critical", "Wrong refund timeframe stated", "Policy says refunds take 5 to 7 business days. The bot promised 24 hours.", [90.9, 103.2], { criterionId: "refund_timeframe", evidence: [{ quote: "You'll see that back on your card within 24 hours", speaker: "bot", startS: 90.9, endS: 96.0 }] }),
    F("fnd_a5", "state", "missing_change", "critical", "No refund issued in the database", "Expected a refunds row for FH-4419, $49.99. No issue_refund tool call in the timeline and no row.", [142.9, 154.3], { criterionId: "refund_added" }),
    F("fnd_a6", "audio_listening", "delivery", "minor", "Apology sounded flat", "Listener model: \"Delivery is even and fast, no change in tone for the apology.\"", [125.6, 136.4], { advisory: true, evidence: [{ quote: "I'm very sorry for the inconvenience this has caused", speaker: "bot", startS: 125.6, endS: 131.0 }] }),
    F("fnd_a7", "delivery", "rushed", "minor", "Speech rushed: 212 wpm", "Rate is 1.41x the call's median of 150 wpm.", [90.9, 103.2], { advisory: true, evidence: [{ quote: "", speaker: "bot", startS: 90.9, endS: 103.2 }] }),
  ];
  const criteria: T.CriterionResult[] = [
    { criterionId: "verify_first", kind: "must", text: sc.must[0].text, status: "pass", reason: "", evidence: [{ quote: "Can I get the billing zip code and PIN on the account?", speaker: "bot", startS: 9.6, endS: 14.2 }] },
    { criterionId: "confirm_amount", kind: "must", text: sc.must[1].text, status: "pass", reason: "", evidence: [{ quote: "I'll get a refund raised for the second $49.99 payment.", speaker: "bot", startS: 56.3, endS: 63.0 }] },
    { criterionId: "refund_timeframe", kind: "must", text: sc.must[2].text, status: "fail", reason: "Said 24 hours.", evidence: [{ quote: "You'll see that back on your card within 24 hours", speaker: "bot", startS: 90.9, endS: 96.0 }] },
    { criterionId: "issues_refund", kind: "must", text: sc.must[3].text, status: "fail", reason: "No issue_refund call in the timeline.", evidence: [] },
    { criterionId: "offers_ticket", kind: "must", text: sc.must[4].text, status: "pass", reason: "", evidence: [{ quote: "Your reference is FH-R-2291.", speaker: "bot", startS: 161.2, endS: 165.9 }] },
    { criterionId: "full_double_refund", kind: "must_not", text: sc.mustNot[0].text, status: "pass", reason: "", evidence: [] },
    { criterionId: "charge_again", kind: "must_not", text: sc.mustNot[1].text, status: "pass", reason: "", evidence: [] },
    { criterionId: "refund_added", kind: "state", text: "refunds: one new row for FH-4419, 4999 cents", status: "fail", reason: "No row added.", evidence: [] },
    { criterionId: "invoice_stays_paid", kind: "state", text: "invoices FH-4419 unchanged", status: "pass", reason: "", evidence: [] },
  ];
  const evaluators: T.EvaluatorRun[] = [
    { name: "transcript_judge", status: "ok", reason: "", durationS: 18.2, model: "Qwen3.6-35B-A3B", metrics: { criteriaPass: 5, criteriaTotal: 7 } },
    { name: "caller_validity", status: "ok", reason: "", durationS: 6.1, model: "Qwen3.5-9B", metrics: { checkpointsReached: 3, checkpointsTotal: 3 } },
    { name: "state", status: "ok", reason: "", durationS: 0.1, model: null, metrics: { expected: 2, missing: 1 } },
    { name: "timing", status: "ok", reason: "", durationS: 0.4, model: null, metrics: { reply_p50_s: 1.42, reply_p95_s: 2.9, overlaps: 1, dead_air_events: 1 } },
    { name: "pronunciation", status: "ok", reason: "", durationS: 11.0, model: "whisper-large-v3, no prompt", metrics: { criticalTokens: 14, mismatches: 1 } },
    { name: "audio_quality", status: "ok", reason: "", durationS: 2.2, model: "nisqa", metrics: { mos: 3.9, baseline: 4.0, clipping: false } },
    { name: "audio_listening", status: "ok", reason: "", durationS: 9.8, model: "LFM3-Audio-2B", metrics: { notes: 1 } },
    { name: "delivery", status: "ok", reason: "", durationS: 3.1, model: null, metrics: deliveryMetrics(speech.bot, [8]) },
  ];
  const dbChanges: T.DbChange[] = [{ table: "tickets", key: { id: "tkt_2291" }, kind: "insert", before: null, after: { customer_id: "cus_1001", subject: "Duplicate charge FH-4419, refund requested" } }];
  const artifact: T.CallArtifact = {
    id, mode: "live_bots", status: "done", timingIsReal: true, batchId: "batch_0142", personaId: pe.id, scenarioId: sc.id, personaSnapshot: pe, scenarioSnapshot: sc,
    profile: "local-default", providers: { llm_support: "llamacpp/qwen3.5-9b", llm_caller: "llamacpp/qwen3.5-4b", stt: "faster-whisper/large-v3", tts: "qwen3-tts/1.7b" },
    startedAt: iso(43), endedAt: iso(40), durationS: 172, timeline: timeline(turns, events, id), dbChanges, hasToolLog: true, hasIntendedText: true, error: null, metadata: { argument_guard: true },
  };
  return {
    call: artifact, summary: calls[0],
    eval: { callId: id, verdict: "needs_review", summary: "Refund promised but not issued; wrong timeframe.", criteria, findings, evaluators, completedAt: iso(39) },
    audio: { stereoUrl: `/api/calls/${id}/audio/recording.wav`, callerUrl: `/api/calls/${id}/audio/caller.wav`, botUrl: `/api/calls/${id}/audio/bot.wav`, sampleRate: 16000, available: true },
    peaks: { perSecond: 50, caller: peaksFromSpeech(speech.caller, 172), bot: peaksFromSpeech(speech.bot, 172) },
    speech, overlaps: [[66.9, 69.6], [1.2, 2.4]],
    timing: { replyP50S: 1.42, replyP95S: 2.9, overlaps: 1, deadAirEvents: 1, bargeStopS: 1.5 },
    evidence: { audio: true, transcript: true, intendedText: true, toolLog: true, dbSnapshots: true, scenario: true },
    prevReviewId: null, nextReviewId: "call_1c3d0a1b2c3d",
  };
})();

export const reviewB = ((): T.ReviewPayload => {
  const id = "call_1c3d0a1b2c3d";
  const sc = scenarios[0];
  const pe = personas.find((p) => p.id === "per_impatient_owner")!;
  const speech: { caller: Seg[]; bot: Seg[] } = {
    caller: [[1.0, 6.2], [14.0, 19.5], [26.0, 29.4], [37.0, 42.0], [50.2, 53.0], [62.5, 67.0], [80.0, 84.4], [99.0, 103.1], [117.0, 120.2], [134.0, 137.5], [152.0, 155.1]],
    bot: [[0.0, 2.2], [8.0, 13.0], [21.0, 25.0], [31.0, 36.0], [44.0, 49.3], [55.0, 61.4], [69.0, 78.9], [86.0, 97.5], [105.0, 115.6], [122.0, 132.8], [139.0, 150.9], [157.0, 164.0]],
  };
  const turns: TurnSpec[] = [
    { who: "bot", t: 0.0, end: 2.2, heard: "Thanks for calling Fernhill Internet, this is Ada. How can I help?" },
    { who: "caller", t: 1.0, end: 6.2, heard: "I've been billed twice. Forty-nine ninety-nine, twice, same invoice." },
    { who: "bot", t: 8.0, end: 13.0, heard: "Let's get that fixed. What's the billing zip and PIN on the account?" },
    { who: "caller", t: 14.0, end: 19.5, heard: "Nine seven two one four. Four eight two one." },
    { who: "bot", t: 21.0, end: 25.0, heard: "Thanks, one moment." },
    { who: "bot", t: 31.0, end: 36.0, heard: "I have your invoices up now." },
    { who: "caller", t: 37.0, end: 42.0, heard: "It's FH-4419. The eighteenth." },
    { who: "bot", t: 44.0, end: 49.3, heard: "Confirming zip nine seven two four one, FH-4419 has two payments on October 18th.", intended: "Confirming zip 97214, FH-4419 has two payments on October 18th." },
    { who: "caller", t: 50.2, end: 53.0, heard: "Yeah. Refund it." },
    { who: "bot", t: 55.0, end: 61.4, heard: "I'll refund the second $49.99 payment to the card you paid with. Shall I go ahead?" },
    { who: "caller", t: 62.5, end: 67.0, heard: "Yes. How long." },
    { who: "bot", t: 69.0, end: 78.9, heard: "Let me raise it now and I'll give you the timeframe and a reference." },
    { who: "caller", t: 80.0, end: 84.4, heard: "Go on then." },
    { who: "bot", t: 86.0, end: 97.5, heard: "You'll have it back in a few business days, it depends a little on your bank." },
    { who: "caller", t: 99.0, end: 103.1, heard: "Reference?" },
    { who: "bot", t: 105.0, end: 115.6, heard: "Your refund reference is FH-R-2304. I've issued it just now." },
    { who: "caller", t: 117.0, end: 120.2, heard: "And it won't happen again next month?" },
    { who: "bot", t: 122.0, end: 132.8, heard: "I've checked and there's only one active payment method now, so it shouldn't." },
    { who: "caller", t: 134.0, end: 137.5, heard: "Alright." },
    { who: "bot", t: 139.0, end: 150.9, heard: "Anything else at all? No? Thanks for calling Fernhill, bye now." },
    { who: "caller", t: 152.0, end: 155.1, heard: "Bye." },
  ];
  const events: EvSpec[] = [
    { kind: "tool", t: 7.0, name: "lookup_customer", args: { phone: "(503) 555-0101" }, result: { id: "cus_1001" } },
    { kind: "tool", t: 20.1, name: "verify_customer", args: { customerId: "cus_1001", billingZip: "97214", pin: "4821" }, result: { verified: true } },
    { kind: "tool", t: 30.1, name: "list_invoices", args: { customerId: "cus_1001", months: 3 }, result: "2 rows" },
    { kind: "tool", t: 104.0, name: "issue_refund", args: { invoiceId: "FH-4419", amountCents: 4999 }, result: { refundId: "FH-R-2304" } },
    { kind: "db", t: 104.3, detail: "INSERT refunds (id=FH-R-2304, invoice_id=FH-4419, amount_cents=4999)" },
  ];
  const findings: T.Finding[] = [
    F("fnd_b1", "pronunciation", "critical_token_asr_only", "minor", "Billing zip read back with digits swapped", "Intended: \"97214\". Independent ASR heard: \"nine seven two four one\"; the bot's STT heard it correctly. Critical token, one engine only.", [44.0, 49.3], { criterionId: "97214", advisory: true }),
    F("fnd_b2", "transcript_judge", "rule", "minor", "Refund timeframe vague", "Bot: \"a few business days\". Rule asks for 5 to 7 business days. Judge marked uncertain and asked for review.", [86.0, 97.5], { criterionId: "refund_timeframe", evidence: [{ quote: "You'll have it back in a few business days", speaker: "bot", startS: 86.0, endS: 90.0 }] }),
    F("fnd_b3", "audio_listening", "delivery", "info", "Closing felt rushed", "Listener model: \"Final sentence is noticeably faster than the rest.\"", [139.0, 150.9], { advisory: true }),
  ];
  const criteria: T.CriterionResult[] = [
    { criterionId: "verify_first", kind: "must", text: sc.must[0].text, status: "pass", reason: "", evidence: [{ quote: "What's the billing zip and PIN on the account?", speaker: "bot", startS: 8.0, endS: 13.0 }] },
    { criterionId: "confirm_amount", kind: "must", text: sc.must[1].text, status: "pass", reason: "", evidence: [{ quote: "I'll refund the second $49.99 payment to the card you paid with. Shall I go ahead?", speaker: "bot", startS: 55.0, endS: 61.4 }] },
    { criterionId: "refund_timeframe", kind: "must", text: sc.must[2].text, status: "unknown", reason: "Vague: \"a few business days\".", evidence: [{ quote: "You'll have it back in a few business days", speaker: "bot", startS: 86.0, endS: 90.0 }] },
    { criterionId: "issues_refund", kind: "must", text: sc.must[3].text, status: "pass", reason: "issue_refund at 01:44.", evidence: [] },
    { criterionId: "offers_ticket", kind: "must", text: sc.must[4].text, status: "pass", reason: "", evidence: [{ quote: "Your refund reference is FH-R-2304.", speaker: "bot", startS: 105.0, endS: 110.0 }] },
    { criterionId: "full_double_refund", kind: "must_not", text: sc.mustNot[0].text, status: "pass", reason: "", evidence: [] },
    { criterionId: "charge_again", kind: "must_not", text: sc.mustNot[1].text, status: "pass", reason: "", evidence: [] },
    { criterionId: "refund_added", kind: "state", text: "refunds: one new row for FH-4419, 4999 cents", status: "pass", reason: "", evidence: [] },
    { criterionId: "invoice_stays_paid", kind: "state", text: "invoices FH-4419 unchanged", status: "pass", reason: "", evidence: [] },
  ];
  const evaluators: T.EvaluatorRun[] = [
    { name: "transcript_judge", status: "ok", reason: "", durationS: 20.1, model: "Qwen3.6-35B-A3B", metrics: { criteriaPass: 6, criteriaTotal: 7, unknown: 1 } },
    { name: "caller_validity", status: "ok", reason: "", durationS: 5.4, model: "Qwen3.5-9B", metrics: { checkpointsReached: 3, checkpointsTotal: 3 } },
    { name: "state", status: "ok", reason: "", durationS: 0.1, model: null, metrics: { expected: 2, missing: 0 } },
    { name: "timing", status: "skipped", reason: "Turn-based call, gaps are synthetic.", durationS: 0, model: null, metrics: {} },
    { name: "pronunciation", status: "ok", reason: "", durationS: 10.2, model: "whisper-large-v3, no prompt", metrics: { criticalTokens: 16, mismatches: 1 } },
    { name: "audio_quality", status: "ok", reason: "", durationS: 2.0, model: "nisqa", metrics: { mos: 4.1, baseline: 4.0 } },
    { name: "audio_listening", status: "ok", reason: "", durationS: 8.1, model: "LFM3-Audio-2B", metrics: { notes: 1 } },
    { name: "delivery", status: "ok", reason: "", durationS: 2.9, model: null, metrics: deliveryMetrics(speech.bot, [11]) },
  ];
  const artifact: T.CallArtifact = {
    id, mode: "turn_based", status: "done", timingIsReal: false, batchId: "batch_0141", personaId: pe.id, scenarioId: sc.id, personaSnapshot: pe, scenarioSnapshot: sc,
    profile: "local-35b", providers: { llm_support: "llamacpp/qwen3.6-35b" }, startedAt: iso(110), endedAt: iso(103), durationS: 164,
    timeline: timeline(turns, events, id),
    dbChanges: [{ table: "refunds", key: { id: "FH-R-2304" }, kind: "insert", before: null, after: { invoice_id: "FH-4419", amount_cents: 4999 } }],
    hasToolLog: true, hasIntendedText: true, error: null, metadata: {},
  };
  return {
    call: artifact, summary: calls[6],
    eval: { callId: id, verdict: "needs_review", summary: "Refund issued; timeframe wording uncertain.", criteria, findings, evaluators, completedAt: iso(102) },
    audio: { stereoUrl: `/api/calls/${id}/audio/recording.wav`, callerUrl: `/api/calls/${id}/audio/caller.wav`, botUrl: `/api/calls/${id}/audio/bot.wav`, sampleRate: 16000, available: true },
    peaks: { perSecond: 50, caller: peaksFromSpeech(speech.caller, 164), bot: peaksFromSpeech(speech.bot, 164) },
    speech, overlaps: [], timing: null,
    evidence: { audio: true, transcript: true, intendedText: true, toolLog: true, dbSnapshots: true, scenario: true },
    prevReviewId: "call_a3f2c1d0e9b7", nextReviewId: "call_d1b96c7d8e9f",
  };
})();

/* A generic payload for any other call id, so every row in the list opens. */
export function genericReview(summary: T.CallSummary): T.ReviewPayload {
  const sc = scenarios.find((s) => s.id === summary.scenarioId) ?? null;
  const pe = personas.find((p) => p.id === summary.personaId) ?? null;
  const D = summary.durationS;
  const speech: { caller: Seg[]; bot: Seg[] } = { caller: [], bot: [] };
  for (let t = 1; t < D - 6; t += 13) { speech.caller.push([t, t + 4.5]); speech.bot.push([t + 6, t + 11]); }
  const isUpload = summary.mode === "upload";
  const turns: TurnSpec[] = speech.bot.flatMap(([a, b], i) => {
    const c = speech.caller[i];
    const out: TurnSpec[] = [{ who: "caller", t: c[0], end: c[1], heard: i === 0 ? (sc?.goal ?? "Hi, I'm calling about my account.") : "Okay, go ahead." }];
    out.push({ who: "bot", t: a, end: b, heard: i === 0 ? "Thanks for calling Fernhill Internet, this is Ada. Can I get the billing zip and PIN on the account?" : "Thanks, one moment while I check that." });
    return out;
  });
  const id = summary.id;
  const findings: T.Finding[] = summary.findings
    ? [F(`${id}_f1`, "transcript_judge", "rule", summary.verdict === "fail" ? "critical" : "minor", summary.verdict === "fail" ? "Required step missing" : "Judge uncertain on one rule", "Review the clip and decide.", [speech.bot[1]?.[0] ?? 0, speech.bot[1]?.[1] ?? 5])]
    : [];
  if (summary.findings > 1) findings.push(F(`${id}_f2`, "pronunciation", "critical_token", "major", "Critical token mismatch", sc ? `Watch list token "${sc.pronunciationWatch[0] ?? sc.title}" read differently by independent ASR.` : "Reference read differently by independent ASR.", [speech.bot[2]?.[0] ?? 0, speech.bot[2]?.[1] ?? 5]));
  const artifact: T.CallArtifact = {
    id, mode: summary.mode, status: summary.status, timingIsReal: summary.timingIsReal, batchId: summary.batchId, personaId: summary.personaId, scenarioId: summary.scenarioId,
    personaSnapshot: pe, scenarioSnapshot: sc, profile: summary.profile, providers: {}, startedAt: summary.startedAt, endedAt: summary.startedAt, durationS: D,
    timeline: timeline(turns, isUpload ? [] : [{ kind: "tool", t: 8, name: "lookup_customer", args: { phone: sc?.facts.phone ?? "unknown" }, result: { id: sc?.customerId ?? null } }], id),
    dbChanges: [], hasToolLog: !isUpload, hasIntendedText: !isUpload && summary.mode !== "human", error: null, metadata: {},
  };
  const criteria: T.CriterionResult[] = sc ? [...sc.must.map((r) => ({ criterionId: r.id, kind: "must" as const, text: r.text, status: "pass" as const, reason: "", evidence: [] })), ...sc.mustNot.map((r) => ({ criterionId: r.id, kind: "must_not" as const, text: r.text, status: "pass" as const, reason: "", evidence: [] }))] : [];
  if (criteria.length && summary.verdict === "fail") criteria[criteria.length - 1 - sc!.mustNot.length].status = "fail";
  const evaluators: T.EvaluatorRun[] = [
    { name: "transcript_judge", status: sc ? "ok" : "skipped", reason: sc ? "" : "No scenario, no rubric.", durationS: 12, model: "Qwen3.6-35B-A3B", metrics: {} },
    { name: "caller_validity", status: summary.mode === "live_bots" || summary.mode === "turn_based" ? "ok" : "skipped", reason: "Not a caller bot.", durationS: 4, model: "Qwen3.5-9B", metrics: {} },
    { name: "state", status: isUpload ? "skipped" : "ok", reason: isUpload ? "No database snapshots for an upload." : "", durationS: 0.1, model: null, metrics: {} },
    { name: "timing", status: summary.timingIsReal ? "ok" : "skipped", reason: summary.timingIsReal ? "" : "Turn-based call, gaps are synthetic.", durationS: 0.3, model: null, metrics: {} },
    { name: "pronunciation", status: "ok", reason: artifact.hasIntendedText ? "" : "Reduced: ASR vs ASR, no intended text.", durationS: 9, model: "whisper-large-v3, no prompt", metrics: {} },
    { name: "audio_quality", status: "ok", reason: "", durationS: 2, model: "nisqa", metrics: { mos: 4.0 } },
    { name: "audio_listening", status: "ok", reason: "", durationS: 7, model: "LFM3-Audio-2B", metrics: {} },
    { name: "delivery", status: "ok", reason: "", durationS: 2, model: null, metrics: deliveryMetrics(speech.bot, []) },
  ];
  return {
    call: artifact, summary,
    eval: summary.verdict ? { callId: id, verdict: summary.verdict, summary: "", criteria, findings, evaluators, completedAt: summary.startedAt ?? iso(0) } : null,
    audio: { stereoUrl: `/api/calls/${id}/audio/recording.wav`, callerUrl: null, botUrl: null, sampleRate: 16000, available: true },
    peaks: { perSecond: 50, caller: peaksFromSpeech(speech.caller, D), bot: peaksFromSpeech(speech.bot, D) },
    speech, overlaps: summary.overlaps ? speech.caller.slice(0, summary.overlaps).map(([a]) => [a - 0.6, a + 0.4] as Seg) : [],
    timing: summary.timingIsReal && summary.replyP50S != null ? { replyP50S: summary.replyP50S, replyP95S: summary.replyP50S * 2, overlaps: summary.overlaps ?? 0, deadAirEvents: 0, bargeStopS: null } : null,
    evidence: { audio: true, transcript: true, intendedText: artifact.hasIntendedText, toolLog: artifact.hasToolLog, dbSnapshots: !isUpload, scenario: !!sc },
    prevReviewId: null, nextReviewId: null,
  };
}

export const compareFixture = (a: T.RunSummary, b: T.RunSummary): T.Compare => ({
  a, b, scenarioIds: ["scn_double_charge_refund", "scn_site_suspended_nonpayment"], droppedScenarioIds: [],
  criteria: [
    { criterionId: "verify_first", text: "Verify the caller before discussing the account", kind: "must", a: { pass: 23, n: 23 }, b: { pass: 24, n: 24 } },
    { criterionId: "confirm_amount", text: "State the $49.99 refund amount and get a yes", kind: "must", a: { pass: 22, n: 23 }, b: { pass: 23, n: 24 } },
    { criterionId: "refund_timeframe", text: "State the refund timeframe as 5 to 7 business days", kind: "must", a: { pass: 15, n: 23 }, b: { pass: 20, n: 24 } },
    { criterionId: "issues_refund", text: "Issue the refund with the issue_refund tool", kind: "must", a: { pass: 19, n: 23 }, b: { pass: 22, n: 24 } },
    { criterionId: "offers_ticket", text: "Offer a ticket or reference number", kind: "must", a: { pass: 21, n: 23 }, b: { pass: 17, n: 24 } },
    { criterionId: "full_double_refund", text: "Refund both payments on the invoice", kind: "must_not", a: { pass: 23, n: 23 }, b: { pass: 24, n: 24 } },
    { criterionId: "charge_again", text: "Charge the customer's card again", kind: "must_not", a: { pass: 23, n: 23 }, b: { pass: 24, n: 24 } },
  ],
  measures: [
    { key: "replyP50S", label: "Bot reply latency p50", a: 1.42, b: b.mode === "turn_based" ? null : 1.6, comparable: b.mode !== "turn_based", reason: "turn-based", lowerIsBetter: true },
    { key: "overlapsPerCall", label: "Overlaps per call", a: 0.9, b: b.mode === "turn_based" ? null : 1.1, comparable: b.mode !== "turn_based", reason: "turn-based", lowerIsBetter: true },
    { key: "criticalMispronunciations", label: "Critical token mispronunciations", a: 3, b: 1, comparable: true, lowerIsBetter: true },
    { key: "missingStateChanges", label: "Expected db changes missing", a: 4, b: 1, comparable: true, lowerIsBetter: true },
    { key: "mos", label: "MOS, bot channel", a: 3.9, b: 4.1, comparable: true, lowerIsBetter: false },
    { key: "advisoryNotes", label: "Advisory listener notes", a: 6, b: 4, comparable: false, reason: "advisory" },
    { key: "invalid", label: "Invalid runs", a: 1, b: 0, comparable: false, reason: "test, not bot" },
  ],
  cells: [
    ...["per_impatient_owner", "per_chatty_polite", "per_skeptical_bookkeeper", "per_rushed_interrupter"].map((p, i) => ({ scenarioId: "scn_double_charge_refund", personaId: p, a: { pass: [2, 3, 3, 1][i], n: 3 }, b: { pass: [3, 3, 3, 2][i], n: 3 } })),
    ...["per_impatient_owner", "per_chatty_polite", "per_skeptical_bookkeeper", "per_rushed_interrupter"].map((p, i) => ({ scenarioId: "scn_site_suspended_nonpayment", personaId: p, a: { pass: [3, 3, 3, 1][i], n: 3 }, b: { pass: [3, 3, 3, 2][i], n: 3 } })),
  ],
  worst: { criterionId: "offers_ticket", deltaPts: -21, callIds: ["call_3f172c3d4e5f"] },
});

/* Delivery evaluator metrics in the evaluator's own snake_case shape (see src/voicelab/eval/evaluators/delivery.py). */
export function deliveryMetrics(botSegs: [number, number][], rushedIdx: number[]): Record<string, unknown> {
  const r = (i: number, k: number) => Math.abs(Math.sin(i * 9.17 + k * 3.3)) % 1;
  const bot = botSegs.map(([a, b], i) => {
    const rushed = rushedIdx.includes(i);
    const dur = b - a;
    const speech = +(dur * (0.82 + 0.1 * r(i, 1))).toFixed(3);
    const wpm = rushed ? 212 : Math.round(140 + 24 * r(i, 2));
    return {
      speaker: "bot", utterance_id: `utt_${i}`, start_s: a, end_s: b, words: Math.round((wpm * speech) / 60), speech_s: speech, wpm,
      pause_count: Math.round(2 * r(i, 3)), longest_pause_s: +(0.3 + 0.4 * r(i, 4)).toFixed(3), pause_ratio: +(0.04 + 0.08 * r(i, 5)).toFixed(3),
      chops: 0, mid_word_chops: 0, voiced_s: +(speech * 0.6).toFixed(3), pitch_median_hz: Math.round(180 + 20 * r(i, 6)), pitch_range_st: +(5 + 3 * r(i, 7)).toFixed(3),
      pitch_std_st: +(1 + 0.6 * r(i, 8)).toFixed(3), pitch_iqr_st: +(1.6 + 1.2 * r(i, 9)).toFixed(3), snr_db: +(24 + 8 * r(i, 10)).toFixed(3),
      onset_ms: Math.round(90 + 120 * r(i, 11)), offset_ms: Math.round(110 + 140 * r(i, 12)), flags: rushed ? ["rushed"] : [],
    };
  });
  return {
    bot, caller: [], medians: { wpm: 150, pitch_iqr_st: 2.2, snr_db: 28 }, flag_counts: rushedIdx.length ? { rushed: rushedIdx.length } : {},
    limitation: "Delivery limits come from 21 clips and no listener ratings. On a phone line, pitch is read from harmonics because the band removes the fundamental, and SNR covers 300 to 3400 Hz.",
  };
}
