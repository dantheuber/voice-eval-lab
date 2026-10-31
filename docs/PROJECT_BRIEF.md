# voice-eval-lab project brief

Every agent working on this repo reads this first. It records what the user asked for, what has been decided, and how the work is organized. If you need to deviate from a decision here, say so in your report instead of silently changing course.

## What we are building

A local, self-hosted experiment: a platform for testing a voice AI customer support bot.

1. **A support bot under test.** A basic voice customer support agent for a mock internet company. It helps callers with account and billing problems.
2. **Bot-to-bot voice testing.** A caller bot phones the support bot and behaves according to a configurable **persona** and **scenario** (the issue the persona is calling about). Real audio crosses the line in both directions. Each side only knows what it heard.
3. **Post-call evaluation that listens to the recording.** Evaluators work from the recorded audio, not only from internal logs.
4. **Human calls.** The user talks to the support bot through a browser microphone. The call is recorded and evaluated post-call the same way. Talk-over and interruption must work live and be measurable afterward.
5. **A dashboard** for reviewing calls and evals, creating personas and test scenarios, launching runs, and making mic calls.

All models must run locally on the user's PC. Every model and service sits behind a replaceable interface so that any piece can be swapped for AWS Bedrock or another provider with a config change and at most one small adapter.

Treat this as a prototype someone will actually use, not a demo. It must start with one command and work end to end with real models.

## Confirmed decisions

| Topic | Decision |
| --- | --- |
| Bot-to-bot timing | Live mode if benchmarks show the hardware keeps up. Otherwise turn-based, slower than real time, with real audio still crossing the line. Every call is labeled with its mode. |
| Mic calls | Always live and full-duplex, with barge-in. Bot audio stops when the human talks over it, and queued audio is discarded. The bot's context reflects only what was actually heard. Overlaps are measured post-call. |
| Echo | Headphones give the cleanest result, but speakers must not cause constant false interruptions. Use browser echo cancellation and handle the rest server-side. |
| Access | Serve on localhost and on the LAN (host IP 192.168.0.75) with a self-signed cert, since browsers only allow the mic on localhost or HTTPS. |
| Machine sharing | Do not touch the user's Ollama service or whisper-sandbox containers. They hold no GPU devices. Run our own model servers on our own ports. Up to about 40 GB of model downloads is fine. |
| Licenses | No non-commercial weights, so the design can move to a company later. Record every model's code and weights license. LFM3 is allowed even if its terms are conditional; flag the terms. |
| AWS | No Bedrock access exists on this machine. Build Bedrock adapters anyway, unit test them against mocks, and label them clearly as untested against live AWS. |
| Mock company | **Fernhill Internet**, a fictional company selling email hosting, website hosting, domains, and SSL. Backed by a SQLite database with real tool calls, so evals can assert on what the bot actually did. |
| Repo | `dantheuber/voice-eval-lab`, private, at `~/projects/voice-eval-lab`. Work on branch `prototype`. |
| Argument guard | Keep the bot-side check on tool arguments, on by default, with `support_bot.argument_guard` to turn it off. It uses only what the bot heard. It must hide nothing: every stopped call is a failed `ToolCallEvent`, and evals report it. |
| Scope | English only, single user, no login. No real phone calls: a simulated phone line (8 kHz mu-law, band-limit, noise, packet loss) sits behind a transport interface so real SIP can be added later. Outside recordings can be uploaded for post-call eval. |

## This machine

- Framework Desktop, Ryzen AI MAX+ 395, 16 cores / 32 threads
- 122 GB unified memory, about 89 GB free. GTT is sized for up to 124 GB, so the iGPU can address nearly all of it.
- Radeon 8060S iGPU (`gfx1151`). Vulkan works. ROCm 7.2 PyTorch wheels work; ROCm 7.1 segfaults on this GPU. **No CUDA.** NPU present at `/dev/accel0`, unused.
- User is in the `render` and `video` groups. No passwordless sudo, so nothing may require `sudo` or apt. `ffmpeg` and `sox` are not installed; use a static binary or a Python package inside the project.
- Python 3.12 available through uv. Node 26, pnpm 11. Docker available without GPU runtimes.
- `~/projects/cc-tts` is a working example of Qwen3-TTS on ROCm 7.2 with uv. Read its README and `pyproject.toml` for the Strix Halo gotchas. `~/projects/whisper-bot/services` has working local STT/TTS servers worth reading.
- Already cached in `~/.cache/huggingface/hub`: LFM3-Audio-2B-GGUF (Q8), Qwen3-TTS 0.6B and 1.7B (Base and CustomVoice), kyutai pocket-tts, Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-35B-A3B, gpt-oss-20b, Qwen3-32B-AWQ. The `hf` CLI is at `~/.local/bin/hf`.
- Ollama (do not modify) serves `qwen3.8:27b-mtp-q8_0`, `gemma4:12b`, and others at `172.17.0.1:11434`, Vulkan, one model at a time.

## Architecture direction

Monorepo. Python 3.12 backend managed by uv under `src/voicelab`. React + TypeScript + Vite frontend under `web/`, managed with pnpm. SQLite plus files on disk under `data/` (gitignored).

**Pipecat** (pinned, BSD-2) is the runtime for live audio pipelines: VAD, turn detection, interruption handling, WebRTC or WebSocket transport, and stereo call recording. Do not reinvent barge-in.

Target layout:

```
config/                model profiles (YAML), one file per provider stack
seeds/                 personas, scenarios, company data
scripts/               setup, model download, benchmarks
src/voicelab/
  core/                shared types, audio utilities, config, provider registry
  providers/           interfaces and adapters (local, OpenAI-compatible, Bedrock)
  company/             Fernhill database, tools, policies
  agents/              support bot and caller bot logic
  telephony/           simulated phone line and transports
  engine/              call orchestration: live and turn-based
  eval/                post-call pipeline and evaluators
  store/               persistence
  api/                 FastAPI app, REST and WebSocket
web/                   dashboard
tests/
docs/
```

### Provider seams

Everything that touches a model goes through a small interface selected by config. Adapters never leak into engine, agent, or eval code.

- `STT`: audio in, transcript with word timestamps and confidence out
- `TTS`: text and voice in, streaming audio out
- `LLM`: chat with tools, streaming, structured output. An OpenAI-compatible adapter covers llama.cpp, Ollama, and vLLM. Bedrock uses Converse.
- `RealtimeVoice`: speech-to-speech session, for LFM3 `/v1/realtime` and Nova Sonic
- `AudioAnalyst`: an audio-capable model answers questions about a clip

Pipecat services wrap these interfaces so one adapter works in live pipelines, turn-based calls, and post-call evals.

### One call artifact

Every call, however it was produced, ends as the same artifact: a stereo recording (caller left, bot right), per-channel files, and an ordered event timeline holding utterances with intended text, heard text, timestamps, tool calls and results, barge-ins, and database changes. Post-call eval takes only this artifact. An uploaded recording is the same artifact with less evidence, and evaluators report what they could not check.

### Post-call evaluation

Evaluators produce findings with a timestamp, severity, evidence, and clip. Planned evaluators:

- **Transcript judge:** pass or fail on each criterion from the scenario rubric, with evidence that has to exist: a quote from the independent transcript, or a cited tool call
- **Caller validity:** did the caller reach the scenario and stay within its facts? A failure is a test problem, not a bot failure, under the verdict rules below
- **State assertions:** diff of the Fernhill database against expected changes
- **Tool attempts:** every tool call the argument guard objected to, checked against what the caller really said. A made-up value that was stopped is advisory, and one that reached a tool fails the call. The README has the measured reason.
- **Timing:** response latency, overlap, dead air, barge-in reaction time
- **Pronunciation:** intended text vs independent ASR of the audio, weighted to critical tokens (names, numbers, amounts, URLs, acronyms), plus a lexicon. Run ASR without domain hints, which can hide mispronunciation.
- **Audio quality:** clipping, loudness, silence, a MOS predictor, compared against a baseline
- **Delivery:** deterministic measurements per utterance, in numpy with no model: speaking rate, pauses, mid-word chops, pitch movement, SNR, and abrupt edges, checked against fixed ranges and the call's own median. Advisory only. Added because LFM3-Audio missed all 5 planted defects in the listener benchmark. On 21 real clips with the same defects, wideband and through the phone line, it caught 105 of 105 within a call and 95 of 105 on fixed limits alone, with 1 false flag on the 21 clean clips (a plosive read as an abrupt onset). The limits have not been checked against listener ratings.
- **Audio listening:** an audio-capable model listens to clips for tone and odd delivery. Advisory only. Research shows these models agree with humans less than 80% of the time, so this flags clips for review and never fails a call on its own. Where it contradicts the delivery measurements, the measurements win and the summary says so.

Each call ends as `pass`, `fail`, `needs_review`, or `invalid`. Humans confirm or dismiss findings in the dashboard, and those verdicts are stored for judge calibration.

Verdict rules, decided from real calls where a bot refunded $189 past its limit and the run came out `invalid`:

- **Commission** means the bot took a forbidden action: a failed `no_change`, a deletion or a write to another customer's account, changing an account before verifying the caller, or a failed `must_not` with a verified quote or tool call. That's `fail` whatever the caller missed, because it really happened.
- **Omission** means the bot didn't do something expected: a failed `must`, or an expected change that didn't happen. Rules and expected changes name the beat they depend on with `needs_beat`. If the caller never reached that beat the bot can't be blamed, so the run is `invalid`. It's `fail` only when the caller did reach it. A rule with no link is excused by any missed required beat.
- A missed required beat with nothing to blame on the bot is `invalid`, unless the bot ended the call, which makes it a bot failure. A caller hanging up early stays `invalid`.
- A material fact-sheet breach makes the run `invalid`, since the bot acted on made-up data. A commission that came before the breach is the exception: the bot did that on honest data, so the call is `fail`, and the summary mentions the breach. A commission with no known time can't be shown to come first, so it counts as after.
- A rule whose beat nobody could decide goes to `needs_review`.
- The summary says why in plain words, including who ended the call and which failed rules weren't counted against the bot.

### Test design rules

Learned from earlier research for this project:

- Give the caller a fixed fact sheet and ordered beats, because a free-running LLM caller is too cooperative and invents details.
- Track whether a run reached its checkpoints separately from whether the bot passed.
- Isolate state. Each call gets its own copy of the seeded database.
- Repeat scenarios and report pass rates, since both sides are stochastic.

## How the work is organized

- Commit small and often on branch `prototype`. Push when a milestone is done. Conventional, plain commit messages.
- Fable 5.1 designs the dashboard UI/UX. GPT-6.1-sol reviews code at each milestone. Findings get fixed before moving on.
- Write research output to `docs/research/`, design output to `docs/design/`.
- Writing style for all docs: plain, specific, no marketing language, no em dashes. State measured numbers and say what was not verified.
- Never commit model weights, recordings, secrets, or anything under `data/`, `.runtime/`, `models/`.
