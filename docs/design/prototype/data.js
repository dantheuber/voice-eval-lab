/* Realistic sample data for the prototype. Fernhill Internet is a fictional email and web host. */
window.VL = window.VL || {};

VL.profiles = [
  { id: "local-default", label: "local-default", desc: "Qwen3.5-9B bot, Qwen3.5-4B caller, whisper STT, Qwen3-TTS", health: "ok" },
  { id: "local-35b", label: "local-qwen36-35b", desc: "Qwen3.6-35B-A3B bot, same caller and audio stack", health: "ok" },
  { id: "local-lfm3", label: "local-lfm3-realtime", desc: "LFM3-Audio-2B speech-to-speech bot", health: "loading" },
  { id: "bedrock-nova", label: "bedrock-nova (untested)", desc: "Nova Sonic bot, Claude judge. Adapters not tested against live AWS.", health: "down" },
];

VL.personas = [
  { id: "p-okafor", name: "Margaret Okafor", age: 67, temperament: "patient", style: "slow, polite, repeats herself", patience: 5, verbosity: 4, interrupt: 1, voice: "qwen3-tts / Serena (older female, RP)", line: "landline hum, 8 kHz" },
  { id: "p-patel", name: "Dev Patel", age: 34, temperament: "impatient", style: "terse, talks fast, swears mildly when stalled", patience: 1, verbosity: 2, interrupt: 5, voice: "qwen3-tts / Ethan (male, London)", line: "mobile, road noise, 2% packet loss" },
  { id: "p-lindqvist", name: "Rosa Lindqvist", age: 45, temperament: "neutral", style: "precise, reads references aloud, asks for ticket numbers", patience: 3, verbosity: 3, interrupt: 2, voice: "qwen3-tts / Vivian (female, Swedish accent)", line: "VoIP, clean, 1% loss" },
  { id: "p-brierley", name: "Tom Brierley", age: 52, temperament: "frustrated", style: "rambles, goes off topic, raises voice when repeating", patience: 2, verbosity: 5, interrupt: 3, voice: "qwen3-tts / Ryan (male, Yorkshire)", line: "landline, TV in background" },
];

VL.scenarios = [
  { id: "S-01", name: "MX records missing after domain transfer", account: "acct 10482, lindqvist-ceramics.co.uk", tags: ["dns", "email"], limit: 24, must: 5, mustnot: 2 },
  { id: "S-02", name: "IMAP login fails after password reset", account: "acct 20917, okaforfamily.uk", tags: ["email"], limit: 20, must: 4, mustnot: 1 },
  { id: "S-03", name: "Refund for duplicate invoice FH-4419", account: "acct 10482, lindqvist-ceramics.co.uk", tags: ["billing"], limit: 20, must: 5, mustnot: 2 },
  { id: "S-04", name: "Website suspended for non-payment", account: "acct 31250, brierleyplumbing.co.uk", tags: ["billing", "hosting"], limit: 26, must: 6, mustnot: 2 },
  { id: "S-05", name: "SSL certificate expired on shop subdomain", account: "acct 10482, shop.lindqvist-ceramics.co.uk", tags: ["ssl", "hosting"], limit: 18, must: 4, mustnot: 1 },
  { id: "S-06", name: "Cancel email hosting, keep the domain", account: "acct 44871, patel-consult.co.uk", tags: ["billing", "email"], limit: 22, must: 5, mustnot: 3 },
];

VL.runs = [
  { id: "run-0142", when: "2026-10-30 09:17", profile: "local-default", mode: "live", calls: 48, done: 48, pass: 31, fail: 9, review: 6, invalid: 2, label: "S-01 to S-06 x 4 personas x 2" },
  { id: "run-0141", when: "2026-10-30 08:02", profile: "local-35b", mode: "turn", calls: 24, done: 24, pass: 19, fail: 2, review: 3, invalid: 0, label: "S-03, S-04 x 4 personas x 3" },
  { id: "run-0140", when: "2026-10-29 17:40", profile: "local-default", mode: "live", calls: 24, done: 24, pass: 17, fail: 4, review: 2, invalid: 1, label: "S-03, S-04 x 4 personas x 3" },
  { id: "run-0139", when: "2026-10-29 16:05", profile: "local-default", mode: "human", calls: 3, done: 3, pass: 2, fail: 0, review: 1, invalid: 0, label: "mic calls, dan" },
  { id: "run-0138", when: "2026-10-29 14:22", profile: "local-lfm3", mode: "live", calls: 12, done: 12, pass: 4, fail: 6, review: 1, invalid: 1, label: "S-02 x 4 personas x 3" },
  { id: "run-0137", when: "2026-10-29 11:50", profile: "local-default", mode: "upload", calls: 1, done: 1, pass: 0, fail: 0, review: 1, invalid: 0, label: "uploaded recording" },
];

// Scenario pass rates across the last 7 days on local-default
VL.scenarioRates = [
  { id: "S-01", n: 22, pass: 18, fail: 2, review: 1, invalid: 1 },
  { id: "S-02", n: 30, pass: 20, fail: 6, review: 3, invalid: 1 },
  { id: "S-03", n: 34, pass: 21, fail: 8, review: 4, invalid: 1 },
  { id: "S-04", n: 34, pass: 27, fail: 3, review: 3, invalid: 1 },
  { id: "S-05", n: 12, pass: 11, fail: 0, review: 1, invalid: 0 },
  { id: "S-06", n: 16, pass: 9, fail: 4, review: 2, invalid: 1 },
];

VL.calls = [
  { id: "c_1030_0917_a3f2", run: "run-0142", scenario: "S-03", persona: "p-patel", mode: "live", profile: "local-default", verdict: "needs_review", findings: 6, open: 4, dur: 172, latency: 1.42, overlap: 2, rep: 1 },
  { id: "c_1030_0917_b801", run: "run-0142", scenario: "S-03", persona: "p-okafor", mode: "live", profile: "local-default", verdict: "pass", findings: 1, open: 0, dur: 241, latency: 1.31, overlap: 0, rep: 1 },
  { id: "c_1030_0917_c44e", run: "run-0142", scenario: "S-01", persona: "p-lindqvist", mode: "live", profile: "local-default", verdict: "fail", findings: 3, open: 3, dur: 198, latency: 1.55, overlap: 1, rep: 1 },
  { id: "c_1030_0917_d1b9", run: "run-0142", scenario: "S-04", persona: "p-brierley", mode: "live", profile: "local-default", verdict: "needs_review", findings: 4, open: 4, dur: 263, latency: 1.38, overlap: 3, rep: 2 },
  { id: "c_1030_0917_e07c", run: "run-0142", scenario: "S-06", persona: "p-patel", mode: "live", profile: "local-default", verdict: "invalid", findings: 1, open: 0, dur: 61, latency: 1.47, overlap: 0, rep: 1, invalidReason: "Caller gave account 44817 instead of 44871 and the call ended before the scenario started." },
  { id: "c_1030_0917_f2aa", run: "run-0142", scenario: "S-02", persona: "p-okafor", mode: "live", profile: "local-default", verdict: "pass", findings: 0, open: 0, dur: 227, latency: 1.29, overlap: 0, rep: 2 },
  { id: "c_1030_0802_1c3d", run: "run-0141", scenario: "S-03", persona: "p-patel", mode: "turn", profile: "local-35b", verdict: "needs_review", findings: 3, open: 3, dur: 164, latency: null, overlap: null, rep: 2 },
  { id: "c_1030_0802_2e90", run: "run-0141", scenario: "S-04", persona: "p-okafor", mode: "turn", profile: "local-35b", verdict: "pass", findings: 1, open: 0, dur: 255, latency: null, overlap: null, rep: 1 },
  { id: "c_1030_0802_3f17", run: "run-0141", scenario: "S-03", persona: "p-lindqvist", mode: "turn", profile: "local-35b", verdict: "fail", findings: 2, open: 2, dur: 188, latency: null, overlap: null, rep: 3 },
  { id: "c_1029_1605_h001", run: "run-0139", scenario: "S-04", persona: null, mode: "human", profile: "local-default", verdict: "needs_review", findings: 2, open: 2, dur: 143, latency: 1.21, overlap: 4, rep: 1 },
  { id: "c_1029_1422_9a2b", run: "run-0138", scenario: "S-02", persona: "p-brierley", mode: "live", profile: "local-lfm3", verdict: "fail", findings: 5, open: 1, dur: 302, latency: 0.74, overlap: 6, rep: 1 },
  { id: "c_1029_1150_u001", run: "run-0137", scenario: null, persona: null, mode: "upload", profile: "local-default", verdict: "needs_review", findings: 2, open: 2, dur: 311, latency: 1.9, overlap: 1, rep: 1 },
];

VL.personaById = id => VL.personas.find(p => p.id === id);
VL.scenarioById = id => VL.scenarios.find(s => s.id === id);

/* Full review payload for two calls. The first is live, the second is turn-based. */
VL.reviews = {
  "c_1030_0917_a3f2": {
    id: "c_1030_0917_a3f2", run: "run-0142", rep: "1 of 2", scenario: "S-03", persona: "p-patel", mode: "live", profile: "local-default",
    verdict: "needs_review", duration: 172, recordedAt: "2026-10-30 09:21:08", line: "mobile, road noise, 2% packet loss", turns: 14, turnLimit: 20,
    speech: {
      // [start, end] seconds, per channel
      caller: [[1.2, 6.8], [15.4, 21.0], [27.9, 31.2], [38.6, 44.1], [52.0, 54.9], [64.2, 70.5], [68.1, 69.3], [84.0, 89.2], [104.5, 109.0], [120.3, 124.1], [138.0, 141.2], [155.6, 160.0], [166.8, 169.4]],
      bot: [[0.0, 1.0], [9.6, 14.2], [22.4, 27.1], [32.5, 37.6], [45.3, 51.1], [56.3, 63.0], [66.9, 69.6], [72.0, 82.6], [90.9, 103.2], [111.0, 119.0], [125.6, 136.4], [142.9, 154.3], [161.2, 165.9], [170.1, 172.0]],
    },
    overlaps: [[66.9, 69.6], [68.1, 69.3]],
    events: [
      { t: 7.9, kind: "tool", label: "lookup_account", detail: "lookup_account(phone=\"07700 900 412\") -> acct 10482 Lindqvist Ceramics Ltd" },
      { t: 31.6, kind: "tool", label: "get_invoices", detail: "get_invoices(account=10482, months=3) -> [FH-4402 paid 29.00, FH-4419 paid 29.00, FH-4419 paid 29.00 (dup)]" },
      { t: 68.1, kind: "barge", label: "barge-in", detail: "Caller started speaking 1.2 s into bot turn; bot stopped after 1.5 s" },
      { t: 110.2, kind: "db", label: "notes +1", detail: "INSERT notes (account=10482, text=\"Caller reports duplicate charge FH-4419, refund requested\")" },
    ],
    findings: [
      { id: "f1", t: 15.4, end: 22.4, sev: "med", source: "timing", title: "Dead air 2.9 s before bot answered", evidence: "Caller finished at 00:21.0, bot started at 00:22.4 after a 1.4 s gap, then paused again. Threshold 2.0 s.", state: null },
      { id: "f2", t: 45.3, end: 51.1, sev: "high", source: "pronunciation", title: "Invoice number read as \"FH forty-four ninety\"", evidence: "Intended: \"invoice FH-4419\". Independent ASR heard: \"invoice FH forty-four ninety\". Critical token mismatch on a reference number.", intended: "invoice FH-4419 for 29 pounds", heard: "invoice FH forty-four ninety for 29 pounds", state: null },
      { id: "f3", t: 66.9, end: 70.5, sev: "low", source: "timing", title: "Bot kept talking 1.5 s after barge-in", evidence: "Caller interrupted at 01:08.1. Bot audio continued until 01:09.6. Target under 0.5 s.", state: null },
      { id: "f4", t: 90.9, end: 103.2, sev: "high", source: "transcript_judge", title: "Wrong refund timeframe stated", evidence: "Bot: \"You'll see that back on your card within 24 hours.\" Policy: refunds take 5 to 7 business days. Rubric C3.", state: null },
      { id: "f5", t: 142.9, end: 154.3, sev: "high", source: "state", title: "No refund issued in the database", evidence: "Expected issue_refund(invoice=FH-4419) and invoices.FH-4419.status = refund_pending. Actual: no tool call, status still paid.", state: null },
      { id: "f6", t: 125.6, end: 136.4, sev: "adv", source: "audio_listening", title: "Apology sounded flat", evidence: "Listener model: \"Delivery is even and fast, no change in tone for the apology.\" Advisory. Cannot fail the call.", state: null },
    ],
    rubric: [
      { id: "C1", kind: "must", text: "Verifies the caller's identity before discussing billing", result: "pass", quote: "Can I take the postcode on the account?", at: 9.6 },
      { id: "C2", kind: "must", text: "Identifies the duplicate charge on FH-4419", result: "pass", quote: "I can see invoice FH-4419 was taken twice on the 14th.", at: 45.3 },
      { id: "C3", kind: "must", text: "States the correct refund timeframe (5 to 7 business days)", result: "fail", quote: "You'll see that back on your card within 24 hours.", at: 96.0 },
      { id: "C4", kind: "must", text: "Issues the refund with the issue_refund tool", result: "fail", quote: null, note: "No issue_refund call in the timeline." },
      { id: "C5", kind: "must", text: "Offers a ticket or reference number", result: "pass", quote: "Your reference is FH-R-2291.", at: 161.2 },
      { id: "N1", kind: "must_not", text: "Does not offer account credit instead of a refund without asking", result: "pass", quote: null },
      { id: "N2", kind: "must_not", text: "Never reads back a full card number", result: "pass", quote: null },
    ],
    dbdiff: [
      { k: "exp", s: "expected  invoices[FH-4419].status  paid -> refund_pending" },
      { k: "miss", s: "actual    invoices[FH-4419].status  paid              (missing)" },
      { k: "exp", s: "expected  refunds  +1 row (invoice=FH-4419, amount=29.00)" },
      { k: "miss", s: "actual    refunds  no change                           (missing)" },
      { k: "ok", s: "actual    notes    +1 row                              (allowed)" },
    ],
    evaluators: [
      { name: "Transcript judge", status: "done", summary: "5 of 7 criteria pass", model: "Qwen3.6-35B-A3B" },
      { name: "Caller validity", status: "done", summary: "Reached scenario, 3 of 3 checkpoints, facts consistent", model: "Qwen3.5-9B" },
      { name: "State assertions", status: "done", summary: "2 expected changes missing", model: "sqlite diff" },
      { name: "Timing", status: "done", summary: "p50 reply 1.42 s, 1 overlap, 1 dead-air event", model: "measured" },
      { name: "Pronunciation", status: "done", summary: "1 critical token mismatch of 14", model: "whisper-large-v3, no prompt" },
      { name: "Audio quality", status: "done", summary: "MOS 3.9 (baseline 4.0), no clipping", model: "nisqa" },
      { name: "Audio listening", status: "done", summary: "1 advisory note", model: "LFM3-Audio-2B", advisory: true },
    ],
    transcript: [
      { who: "bot", t: 0.0, heard: "Fernhill Internet, this is Ada. How can I help?", intended: "Fernhill Internet, this is Ada. How can I help?" },
      { who: "caller", t: 1.2, heard: "Yeah hi, I've been charged twice this month, I need that sorted today." },
      { who: "event", t: 7.9, kind: "tool", text: "lookup_account(phone=\"07700 900 412\") -> acct 10482 Lindqvist Ceramics Ltd" },
      { who: "bot", t: 9.6, heard: "Sorry about that. Can I take the postcode on the account?", intended: "Sorry about that. Can I take the postcode on the account?" },
      { who: "caller", t: 15.4, heard: "S Y two three, four A F. Lindqvist Ceramics." },
      { who: "bot", t: 22.4, heard: "Thanks. Let me pull up your recent invoices.", intended: "Thanks. Let me pull up your recent invoices.", finding: "f1" },
      { who: "caller", t: 27.9, heard: "It's the one from the fourteenth." },
      { who: "event", t: 31.6, kind: "tool", text: "get_invoices(account=10482, months=3) -> 3 rows" },
      { who: "bot", t: 32.5, heard: "Bear with me one second while that loads.", intended: "Bear with me one second while that loads." },
      { who: "caller", t: 38.6, heard: "Twenty-nine quid, twice. Same day." },
      { who: "bot", t: 45.3, heard: "I can see invoice FH forty-four ninety was taken twice on the 14th, 29 pounds each.", intended: "I can see invoice FH-4419 was taken twice on the 14th, 29 pounds each.", finding: "f2", diff: true },
      { who: "caller", t: 52.0, heard: "Forty-four ninety? No, forty-four nineteen." },
      { who: "bot", t: 56.3, heard: "Apologies, FH-4419. That's the one. I'll get a refund raised for the second payment.", intended: "Apologies, FH-4419. That's the one. I'll get a refund raised for the second payment." },
      { who: "caller", t: 64.2, heard: "Right, and how long is that going to take, because last time" },
      { who: "bot", t: 66.9, heard: "The refund goes back to the", intended: "The refund goes back to the card you paid with.", barge: true, finding: "f3" },
      { who: "caller", t: 68.1, heard: "last time it took weeks.", overlap: true },
      { who: "bot", t: 72.0, heard: "I understand. It goes back to the same card you paid with, and I'll add a note so you're not charged twice again next month.", intended: "I understand. It goes back to the same card you paid with, and I'll add a note so you're not charged twice again next month." },
      { who: "caller", t: 84.0, heard: "You didn't answer. How long." },
      { who: "bot", t: 90.9, heard: "You'll see that back on your card within 24 hours, and I'm sorry again for the hassle.", intended: "You'll see that back on your card within 24 hours, and I'm sorry again for the hassle.", finding: "f4" },
      { who: "caller", t: 104.5, heard: "Fine. Put that in writing on the account." },
      { who: "event", t: 110.2, kind: "db", text: "INSERT notes (account=10482, text=\"Caller reports duplicate charge FH-4419, refund requested\")" },
      { who: "bot", t: 111.0, heard: "Done, I've added a note to the account with today's date.", intended: "Done, I've added a note to the account with today's date." },
      { who: "caller", t: 120.3, heard: "And the refund's actually going through, yeah?" },
      { who: "bot", t: 125.6, heard: "Yes, I'm very sorry for the inconvenience this has caused, it is being processed now.", intended: "Yes, I'm very sorry for the inconvenience this has caused, it is being processed now.", finding: "f6" },
      { who: "caller", t: 138.0, heard: "Okay. Reference number?" },
      { who: "bot", t: 142.9, heard: "The refund is in progress and you'll get an email confirmation shortly. Is there anything else I can help with today?", intended: "The refund is in progress and you'll get an email confirmation shortly. Is there anything else I can help with today?", finding: "f5" },
      { who: "caller", t: 155.6, heard: "A reference. Number. For the refund." },
      { who: "bot", t: 161.2, heard: "Of course. Your reference is FH-R-2291.", intended: "Of course. Your reference is FH-R-2291." },
      { who: "caller", t: 166.8, heard: "Right. Bye." },
      { who: "bot", t: 170.1, heard: "Thanks for calling Fernhill.", intended: "Thanks for calling Fernhill." },
    ],
    timing: { p50: 1.42, p95: 2.9, overlaps: 1, deadAir: 1, bargeReact: 1.5 },
    next: "c_1030_0802_1c3d",
  },
  "c_1030_0802_1c3d": {
    id: "c_1030_0802_1c3d", run: "run-0141", rep: "2 of 3", scenario: "S-03", persona: "p-patel", mode: "turn", profile: "local-35b",
    verdict: "needs_review", duration: 164, recordedAt: "2026-10-30 08:14:40", line: "mobile, road noise, 2% packet loss", turns: 12, turnLimit: 20, wallClock: "6 min 48 s",
    speech: {
      caller: [[1.0, 6.2], [14.0, 19.5], [26.0, 29.4], [37.0, 42.0], [50.2, 53.0], [62.5, 67.0], [80.0, 84.4], [99.0, 103.1], [117.0, 120.2], [134.0, 137.5], [152.0, 155.1]],
      bot: [[0.0, 0.9], [8.0, 13.0], [21.0, 25.0], [31.0, 36.0], [44.0, 49.3], [55.0, 61.4], [69.0, 78.9], [86.0, 97.5], [105.0, 115.6], [122.0, 132.8], [139.0, 150.9], [157.0, 164.0] ],
    },
    overlaps: [],
    events: [
      { t: 7.0, kind: "tool", label: "lookup_account", detail: "lookup_account(phone=\"07700 900 412\") -> acct 10482 Lindqvist Ceramics Ltd" },
      { t: 30.1, kind: "tool", label: "get_invoices", detail: "get_invoices(account=10482, months=3) -> 3 rows" },
      { t: 104.0, kind: "tool", label: "issue_refund", detail: "issue_refund(invoice=\"FH-4419\", amount=29.00) -> refund_id FH-R-2304" },
      { t: 104.3, kind: "db", label: "invoices, refunds", detail: "UPDATE invoices SET status=refund_pending WHERE id=FH-4419; INSERT refunds (FH-R-2304, 29.00)" },
    ],
    findings: [
      { id: "f1", t: 44.0, end: 49.3, sev: "med", source: "pronunciation", title: "Postcode read with wrong letter", evidence: "Intended: \"SY23 4AF\". ASR heard: \"SY23 4AS\". Critical token.", intended: "SY23 4AF", heard: "SY23 4AS", state: null },
      { id: "f2", t: 86.0, end: 97.5, sev: "med", source: "transcript_judge", title: "Refund timeframe vague", evidence: "Bot: \"a few working days\". Rubric C3 asks for 5 to 7 business days. Judge marked uncertain and asked for review.", state: null },
      { id: "f3", t: 139.0, end: 150.9, sev: "adv", source: "audio_listening", title: "Closing felt rushed", evidence: "Listener model: \"Final sentence is noticeably faster than the rest.\" Advisory.", state: null },
    ],
    rubric: [
      { id: "C1", kind: "must", text: "Verifies the caller's identity before discussing billing", result: "pass", quote: "What's the postcode on the account?", at: 8.0 },
      { id: "C2", kind: "must", text: "Identifies the duplicate charge on FH-4419", result: "pass", quote: "FH-4419 has two payments on the 14th.", at: 44.0 },
      { id: "C3", kind: "must", text: "States the correct refund timeframe (5 to 7 business days)", result: "review", quote: "You'll have it back in a few working days.", at: 90.0 },
      { id: "C4", kind: "must", text: "Issues the refund with the issue_refund tool", result: "pass", quote: null, note: "issue_refund at 01:44." },
      { id: "C5", kind: "must", text: "Offers a ticket or reference number", result: "pass", quote: "Your refund reference is FH-R-2304.", at: 105.0 },
      { id: "N1", kind: "must_not", text: "Does not offer account credit instead of a refund without asking", result: "pass", quote: null },
      { id: "N2", kind: "must_not", text: "Never reads back a full card number", result: "pass", quote: null },
    ],
    dbdiff: [
      { k: "ok", s: "actual    invoices[FH-4419].status  paid -> refund_pending   (expected)" },
      { k: "ok", s: "actual    refunds  +1 row (FH-R-2304, 29.00)              (expected)" },
    ],
    evaluators: [
      { name: "Transcript judge", status: "done", summary: "6 of 7 pass, 1 needs review", model: "Qwen3.6-35B-A3B" },
      { name: "Caller validity", status: "done", summary: "Reached scenario, 3 of 3 checkpoints", model: "Qwen3.5-9B" },
      { name: "State assertions", status: "done", summary: "All expected changes present", model: "sqlite diff" },
      { name: "Timing", status: "skipped", summary: "Not measured. Turn-based call, gaps are synthetic.", model: null },
      { name: "Pronunciation", status: "done", summary: "1 critical token mismatch of 16", model: "whisper-large-v3, no prompt" },
      { name: "Audio quality", status: "done", summary: "MOS 4.1, no clipping", model: "nisqa" },
      { name: "Audio listening", status: "done", summary: "1 advisory note", model: "LFM3-Audio-2B", advisory: true },
    ],
    transcript: [
      { who: "bot", t: 0.0, heard: "Fernhill Internet, this is Ada. How can I help?", intended: "Fernhill Internet, this is Ada. How can I help?" },
      { who: "caller", t: 1.0, heard: "I've been billed twice. Twenty-nine pounds, twice, same invoice." },
      { who: "event", t: 7.0, kind: "tool", text: "lookup_account(phone=\"07700 900 412\") -> acct 10482 Lindqvist Ceramics Ltd" },
      { who: "bot", t: 8.0, heard: "Let's get that fixed. What's the postcode on the account?", intended: "Let's get that fixed. What's the postcode on the account?" },
      { who: "caller", t: 14.0, heard: "SY23 4AF." },
      { who: "bot", t: 21.0, heard: "Thanks, one moment.", intended: "Thanks, one moment." },
      { who: "event", t: 30.1, kind: "tool", text: "get_invoices(account=10482, months=3) -> 3 rows" },
      { who: "bot", t: 31.0, heard: "I have your invoices up now.", intended: "I have your invoices up now." },
      { who: "caller", t: 37.0, heard: "It's FH-4419. The fourteenth." },
      { who: "bot", t: 44.0, heard: "Confirming SY23 4AS, FH-4419 has two payments on the 14th.", intended: "Confirming SY23 4AF, FH-4419 has two payments on the 14th.", finding: "f1", diff: true },
      { who: "caller", t: 50.2, heard: "Yeah. Refund it." },
      { who: "bot", t: 55.0, heard: "I'll refund the second payment to the card you paid with.", intended: "I'll refund the second payment to the card you paid with." },
      { who: "caller", t: 62.5, heard: "How long." },
      { who: "bot", t: 69.0, heard: "Let me raise it now and I'll give you the timeframe and a reference.", intended: "Let me raise it now and I'll give you the timeframe and a reference." },
      { who: "caller", t: 80.0, heard: "Go on then." },
      { who: "bot", t: 86.0, heard: "You'll have it back in a few working days, it depends a little on your bank.", intended: "You'll have it back in a few working days, it depends a little on your bank.", finding: "f2" },
      { who: "caller", t: 99.0, heard: "Reference?" },
      { who: "event", t: 104.0, kind: "tool", text: "issue_refund(invoice=\"FH-4419\", amount=29.00) -> FH-R-2304" },
      { who: "event", t: 104.3, kind: "db", text: "UPDATE invoices SET status=refund_pending WHERE id=FH-4419; INSERT refunds (FH-R-2304, 29.00)" },
      { who: "bot", t: 105.0, heard: "Your refund reference is FH-R-2304. I've issued it just now.", intended: "Your refund reference is FH-R-2304. I've issued it just now." },
      { who: "caller", t: 117.0, heard: "And it won't happen again next month?" },
      { who: "bot", t: 122.0, heard: "I've checked the mandate and there's only one active now, so it shouldn't.", intended: "I've checked the mandate and there's only one active now, so it shouldn't." },
      { who: "caller", t: 134.0, heard: "Alright." },
      { who: "bot", t: 139.0, heard: "Anything else at all? No? Thanks for calling Fernhill, bye now.", intended: "Anything else at all? No? Thanks for calling Fernhill, bye now.", finding: "f3" },
      { who: "caller", t: 152.0, heard: "Bye." },
    ],
    timing: null,
    next: "c_1030_0917_d1b9",
  },
};

VL.reviewQueue = [
  { id: "c_1030_0917_a3f2", scenario: "S-03", persona: "Dev Patel", open: 4, why: "wrong refund timeframe, no refund issued" },
  { id: "c_1030_0917_d1b9", scenario: "S-04", persona: "Tom Brierley", open: 4, why: "3 overlaps, judge uncertain on C2" },
  { id: "c_1030_0802_1c3d", scenario: "S-03", persona: "Dev Patel", open: 3, why: "postcode misread, vague timeframe" },
  { id: "c_1029_1605_h001", scenario: "S-04", persona: "you (mic)", open: 2, why: "4 overlaps, pronunciation of brierleyplumbing" },
  { id: "c_1029_1150_u001", scenario: "unknown", persona: "upload", open: 2, why: "reduced evidence, judge only" },
  { id: "c_1030_0802_3f17", scenario: "S-03", persona: "Rosa Lindqvist", open: 2, why: "fail, confirm before counting" },
  { id: "c_1030_0917_c44e", scenario: "S-01", persona: "Rosa Lindqvist", open: 3, why: "MX record read wrong twice" },
];

VL.health = [
  { name: "support bot LLM", model: "Qwen3.5-9B-Instruct (llama.cpp :8101)", state: "ok", detail: "ready, 38 tok/s" },
  { name: "caller LLM", model: "Qwen3.5-4B (llama.cpp :8102)", state: "ok", detail: "ready" },
  { name: "STT", model: "faster-whisper large-v3 (:8110)", state: "ok", detail: "ready, RTF 0.21" },
  { name: "TTS", model: "Qwen3-TTS 1.7B CustomVoice (:8120)", state: "ok", detail: "ready, first byte 410 ms" },
  { name: "judge LLM", model: "Qwen3.6-35B-A3B (llama.cpp :8103)", state: "loading", detail: "loading weights, 22 s" },
  { name: "audio analyst", model: "LFM3-Audio-2B (:8130)", state: "ok", detail: "ready" },
  { name: "MOS predictor", model: "NISQA (in-process)", state: "ok", detail: "ready" },
];
