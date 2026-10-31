# voice-eval-lab

voice-eval-lab phones a voice support bot for a mock internet company, using either a scripted caller bot or your own microphone. After each call it evaluates the recording and reports what the bot got right and wrong, with every model running on your own machine.

## Quick start

```bash
git clone git@github.com:dantheuber/voice-eval-lab.git && cd voice-eval-lab
scripts/setup.sh
./run.sh
```

Then open <http://127.0.0.1:18080>.

- You need Linux x86-64, a Vulkan GPU, [uv](https://docs.astral.sh/uv/), and, for the first dashboard build, node and pnpm. Setup and run need no sudo.
- `scripts/setup.sh` asks before it downloads 24.9 GB. It uses about 31 GB of disk, and `--check` shows what's missing without changing anything.
- `./run.sh --fake` skips the models entirely, for a quick look. It needs no setup.
- `python3 scripts/doctor.py` reports what's wrong with the machine.

[docs/USAGE.md](docs/USAGE.md) is the full walkthrough.

## What you can do

| Task | Where | Notes |
| --- | --- | --- |
| Run bot-to-bot calls live | Run launcher, `/runs/new`, mode `live` | Real audio crosses a simulated phone line in both directions, in real time, and each side only knows what it heard |
| Run bot-to-bot calls turn-based | Run launcher, mode `turn-based` | Slower than real time. Real audio still crosses the line, but timing isn't measured |
| Talk to the bot | Talk to the bot, `/talk` | Live mic call with barge-in. Use headphones. From another machine, use the HTTPS address |
| Review a call | Calls, `/calls`, then one call | Evaluation runs by itself. Waveform, transcript, and findings, all usable from the keyboard |
| Evaluate an outside recording | Upload, `/upload` | Add a scenario, intended text, or a tool log to unlock more checks |
| Build personas | Personas, `/personas` | Temperament, speaking style, voice, and line conditions |
| Build scenarios | Scenarios, `/scenarios` | Fact sheet, ordered beats, rules, and expected database changes. Has an LLM draft and a text-only dry run |
| Compare two runs | Compare, `/compare` | Per-criterion and per-measure differences |
| Swap a model or provider | Models and settings, `/settings`, or a YAML profile | See [docs/ADAPTERS.md](docs/ADAPTERS.md) |

## The stack

`./run.sh` starts `local-default`. All numbers come from [the benchmark doc](docs/research/benchmarks-2026-10.md), measured on a Framework Desktop with a Ryzen AI MAX+ 395, 122 GB of unified memory, and a Radeon 8060S on Vulkan.

| Role | Model | Runs on | Measured |
| --- | --- | --- | --- |
| Live STT | Qwen3-ASR 0.6B Q8_0 | llama.cpp b12017, Vulkan | 146 ms median per clip, 46.6x real time, 29 of 32 critical tokens, 2.6 GB |
| Support and caller LLM | Qwen3.5-4B Q4_K_M | llama.cpp, Vulkan, 2 slots | 92 ms warm first token, 65.5 t/s, 12 of 12 tool sessions from speech, 4.0 GB |
| Bot TTS | Phonon 31m q8 | `ptts`, CPU | 23 ms to first audio, 10.5x, 14 of 15 critical tokens |
| Caller TTS | Pocket TTS 18M | `pocket-tts`, CPU | 29 ms to first audio, 12.4x, 13 of 15 critical tokens |
| VAD and turns | Silero, Smart Turn v3.2 | Pipecat 1.15.0, CPU | Silero 0.6 ms per 32 ms window. Smart Turn was not benchmarked |
| Post-call STT | Whisper large-v3 q5_0 | whisper.cpp v1.9.5, Vulkan | 16.9x, WER 0.006 on human speech, 28 of 32 critical tokens, 1.7 GB |
| Judge | Qwen3.8-27B UD-Q4_K_M | llama.cpp, Vulkan | 13 t/s, 35 to 111 s per transcript, 20 of 20 verdicts on one test transcript, 18.7 GB |

Tool accuracy comes from 2 clips of one synthetic voice, and critical tokens from 12 synthetic clips. Those figures rank the stacks and don't predict accuracy on real callers.

Three stacks are ready to use. `--profile` picks one:

| Profile | Stack | Plain turn, median and max | Tool turn, median and max | Live GPU GB | Tool sessions from speech | LFM license limit |
| --- | --- | --- | --- | --- | --- | --- |
| `local-default` | Qwen3-ASR, Qwen3.5-4B, Phonon | 833, 913 ms | 2001, 2161 ms | 6.6 | 12 of 12 | No |
| `local-fast` | Parakeet, LFM3-1.7B, Phonon | 681, 737 ms | 1123, 1147 ms | 2.4 | 9 of 12 | Yes |
| `local-realtime` | LFM3-Audio-2B speech-to-speech | 528, 537 ms | 621, 624 ms | 3.2 | lookup 12 of 12, invoices 0 of 12, refund 11 of 12 | Yes |

Read those latencies with care:

- Each one runs from the end of caller speech to first bot audio, on 8 kHz mu-law input, and includes 420 to 520 ms of VAD hang.
- They come from the benchmark harness, which ran before the live engine existed. The engine waits a 0.5 s resume window at the end of a turn, which costs about 0.4 s on every turn, so real calls run slower than this table. The final numbers belong in the results section below.
- The `local-default` tool row reads low, because its lookup failed and the turn ended early. A successful Qwen3.5-4B tool turn took about 2.4 s with Parakeet, so plan on that. The bot says a hold line when the first LLM round runs long.
- A 2-minute two-bot call on Parakeet, Qwen3.5-4B, Phonon, and Pocket 7M ran 11 turns at 1462 ms median and 2192 ms max, with no late audio windows. `local-default` itself has no such benchmark.

Compute was never the limit in these tests. In four live two-bot calls the GPU averaged 4 to 23% busy. With everything loaded, `local-default` needs about 27.4 GB of GPU memory, which is the benchmark's per-model figures added up.

## How evals work

Every call, however it was made, ends as the same artifact: a stereo recording with caller on the left and bot on the right, one file per channel, and a timeline of utterances, tool calls, barge-ins, and database changes. Evaluation reads only that artifact.

| Evaluator | What it checks | Effect on the verdict |
| --- | --- | --- |
| `transcribe` | Makes an independent Whisper transcript with no domain hints | Feeds the others |
| `timing` | Reply latency, dead air, overlap, and barge-in reaction | A major finding fails the call. Skipped when timing isn't real |
| `state` | Fernhill database changes against the scenario's expected changes | Fails the call, see below |
| `transcript_judge` | Each rule, with a pass or fail and evidence that has to exist: a quote from the transcript or a cited tool call | Fails the call, or sends it to review when undecided |
| `caller_validity` | Whether the caller reached its checkpoints and kept to its fact sheet | Can make the call invalid, see below |
| `pronunciation` | Intended text against what the independent STT heard, weighted to names, numbers, amounts, URLs, and acronyms | Fails the call only when a second check agrees. Otherwise advisory |
| `audio_quality` | Clipping, loudness, silence, and level jumps | Advisory. No MOS predictor is configured |
| `delivery` | Speaking rate, pauses, mid-word chops, pitch, SNR, and abrupt edges, measured in numpy with no model | Advisory |
| `audio_listening` | An audio model's opinion of each clip | Advisory. Skipped on the local profiles, which have no audio model, because each one tested scored chance |

The verdict comes from fixed rules in `src/voicelab/eval/verdict.py`. No LLM touches it. They tell what the bot did apart from what the caller never reached:

- A **commission** is the bot doing something forbidden: a failed `no_change` expectation, a deletion or a write to another customer's account, changing an account before verifying the caller, or a failed `must not` with a verified quote or tool call. It happened, so it counts whatever the caller went on to miss.
- An **omission** is the bot not doing something expected: a failed `must`, or an expected change that didn't happen. A rule names the beat it depends on with `needs_beat`, and the bot only answers for it if the caller reached that beat. A rule with no link is excused by any missed required beat.

The rules apply in this order, and the first match wins:

| Verdict | When |
| --- | --- |
| `fail` | The caller broke a material fact, but a commission came before it. The bot did that on honest data |
| `invalid` | The caller broke a material fact. Whatever the bot did after that, it did on made-up data, and a commission with no known time counts as after |
| `fail` | A commission |
| `fail` | The bot ended the call before the caller reached a required checkpoint |
| `fail` | An omission of something the caller reached |
| `invalid` | The caller missed a required checkpoint, with nothing above to blame on the bot. This is a test problem, not a bot failure, and the call stays out of pass rates. The summary says who ended the call and which failed rules weren't counted |
| `fail` | A non-advisory finding is major or critical |
| `needs_review` | A rule or a beat it depends on couldn't be decided, an evaluator errored, the caller slipped on a minor fact, advisory findings reached a score of 3, or nothing could be confirmed |
| `pass` | None of the above, and at least one rule was confirmed |

A major advisory finding scores 3 and a minor one scores 1, so one major or three minor findings send a call to review. `info` findings score nothing and stay out of the default list, behind a details button on the review screen. Repeats of one kind of problem roll up into one finding that lists every occurrence and plays the worst. People confirm or dismiss findings on the review screen, and their verdict is stored beside the machine's, never over it.

## Known limits

Accuracy and quality:

- Nobody has listened to the benchmark audio. Voice quality was checked with ASR, which shows the words came out and says nothing about how they sound.
- Human speech in the benchmarks is one LibriSpeech speaker and one JFK clip. Real callers, accents, and real phone lines were never tested. Neither were packet loss and line noise.
- The judge's accuracy rests on one transcript with 5 criteria, run 6 times. That shows the plumbing works.
- Delivery limits come from 21 clips with planted defects. They caught 105 of 105 within a call and 95 of 105 on fixed limits, with 1 false flag on 21 clean clips, and they were never checked against listener ratings.
- No audio model hears delivery. LFM3-Audio, Gemma 4 E2B, and Voxtral all found 0 of 20 planted defects.
- STT reports no usable word confidence. Qwen3-ASR gives none, and Parakeet says 1.0 on every word, so pronunciation confidence comes from Whisper alone.
- Before spoken-id handling, no live two-bot call reached a refund. The fix spells ids phone-style, and came back right 10 of 10 on ten numbers. Live pass rates are in the results section once measured.

Live calls:

- Echo handling was tested on synthetic audio only, never with a real speaker in a real room. Use headphones.
- Barge-in timing through Pipecat was not benchmarked. The browser's jitter buffer holds a little audio past the point the server measures, and that gap is not measured.
- More than one two-sided call at once, and eval running during a live call, were not benchmarked.
- `local-fast`: Parakeet loops on repeated digits, which matters here because spoken ids are the core of the product.
- `local-realtime`: LFM3 fires `get_invoices` before the lookup result exists, has one voice, gives no user transcript, and needs a session rebuild on every barge-in. It can't play the caller.

Scope and setup:

- English only, one user, no login. The HTTPS port listens on the LAN. Host and Origin checks guard it, but anyone on your network can reach the dashboard.
- There are no real phone calls. A simulated line sits behind a transport interface.
- The Bedrock adapters have never run against live AWS, and `amazon.nova-2-5-sonic` could not be confirmed in AWS documentation.
- Tested on one machine: Linux x86-64, Radeon 8060S, RADV from Mesa 25.2.8, Ubuntu 25.10. There is no CUDA, ROCm, or macOS path.
- `scripts/setup.sh` was tested on a clean clone with model weights linked in from an existing install, not with a full 24.9 GB download. Every pinned file was checked against Hugging Face by size and hash.

## Results

End-to-end results with the finished engine:

<!-- RESULTS -->

## Repo layout

```
config/profiles/   model stacks as YAML: local-default, local-fast, local-realtime, fake, bedrock.example
docs/              USAGE, ADAPTERS, LIVE_ENGINE, API, PROJECT_BRIEF, research/, design/
run.sh             the one command
scripts/
  setup.sh         installs model servers and models under .runtime/
  setup/           pins: model manifest, conda toolchain lock, venv freeze
  doctor.py        checks the machine
  phonon_server.py HTTP server around ptts
  bench/           benchmark research code, kept as it ran, excluded from ruff
seeds/             personas, scenarios, Fernhill company data, pronunciation lexicon
src/voicelab/
  agents/          support bot and caller bot
  api/             FastAPI routes
  company/         Fernhill database and tools
  core/            shared types, audio, profiles
  engine/          call orchestration, live and turn-based
  eval/            post-call evaluators and the verdict
  providers/       the five interfaces and their adapters
  services/        settings, model server supervisor, runs, uploads
  store/           SQLite and call artifacts
  telephony/       simulated phone line
tests/             pytest, no models needed unless marked
web/               React dashboard
```

Not in git: `.runtime/` (binaries, models, venvs), `data/` (recordings, database, logs), `certs/`, and `web/dist/`.

## Licenses

The repo has no license of its own yet, and it is private. Everything a profile uses is listed with its terms in [LICENSES.md](LICENSES.md). In short:

- `local-default` has no LFM models, so it carries no revenue limit. Its code is permissive, its Qwen and Phonon weights are Apache-2.0, and Whisper is MIT.
- `local-fast` and `local-realtime` use LFM weights. The LFM Open License v1.0 allows commercial use only under USD 10 million in annual revenue.
- Attribution is required for Pocket weights and the VCTK voices (CC-BY-4.0), and for Parakeet in `local-fast` (CC-BY-4.0).
- Pocket's `cosette` and `jean` presets are non-commercial, and the code refuses them.
- A few installed dependencies carry stronger terms. PyAV wheels bundle GPL codecs and PyPI torch pulls proprietary NVIDIA packages. [LICENSES.md](LICENSES.md) has the details.
