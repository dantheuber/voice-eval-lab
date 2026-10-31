# Voice and speech models for voice-eval-lab, October 2026

Survey date: 2026-10-30. Target machine: Radeon 8060S (`gfx1151`), Vulkan, ROCm 7.2, 122 GB unified memory, no CUDA, no sudo.

Nothing here was installed or run. Every number is reported by its source unless marked otherwise. Benchmarking on the real machine is a separate task.

Evidence labels used below:

- **vendor**: stated by the party that released the model. Treat as an upper bound.
- **source**: read directly in the code or license text, at llama.cpp commit `0ec4027` and Pipecat v1.15.0.
- **local**: measured on this machine in an earlier project.
- **third party**: published by someone other than the releaser.
- **unverified**: not confirmed from a primary source. Test before relying on it.

Date columns show the Hugging Face repo creation date. That can precede the public release: the `LiquidAI/LFM3-Audio-2B` repo was created 2026-09-08 and released 2026-10-27. Sizes are the weights to download, not the repo total, unless marked "repo".

## Main findings

1. LFM3-Audio-2B is the only open speech-to-speech model found with tool calling and a documented non-CUDA path. Vulkan is untested in public: the llama.cpp PRs say "smoke-tested on Mac and CUDA". All of its audio graph ops are listed as Vulkan-supported, with `CONV_2D` partial.
2. llama.cpp `/v1/realtime` has no VAD (`turn_detection` is hard-coded to `null`), no `conversation.item.truncate`, no assistant-message injection, no voice parameter, and no user transcript in conversation sessions. The first and fifth can be handled by adding local VAD and STT. The others conflict with the brief's rule that the bot's context reflects only what was heard.
3. Pipecat's `OpenAIRealtimeLLMService` will not work unmodified. On interruption it sends `conversation.item.truncate`. llama.cpp answers `error: unknown event type`, and Pipecat treats that `error` event as fatal and leaves the receive loop (only two specific OpenAI error codes are exempt). A small dedicated adapter is needed.
4. LFM3-Audio's published tool score is 58.5 on Liquid's internal audio eval, against 72.4 for Gemma 4 E4B on the same eval. A cascade with LFM3-1.7B text is the safer default for a bot with real tool calls. Treat LFM3-Audio as the `RealtimeVoice` candidate to compare, not the assumed winner.
5. Liquid published Vulkan numbers on this exact chip. LFM3-1.7B at Q4_0 prefills at 3,771 tok/s and decodes at 129.6 tok/s on Radeon 8060S. Gemma 4 E4B prefills at 337 tok/s on the same hardware, which would cost seconds on a long system prompt.
6. NVIDIA released NeMo-Speech.cpp, an Apache-2.0 ggml runtime with prebuilt Linux Vulkan archives (v0.3.0, 2026-10-30). It serves Parakeet and Nemotron with word timestamps and per-word `confidence`, plus a realtime WebSocket. It is the most direct non-CUDA STT path. Its VoiceChat S2S mode is CUDA-only.
7. Caller voices have a good answer. Kyutai's `tts-voices/voice-donations` holds 228 verified CC0 voices and `vctk/` adds CC-BY-4.0 voices with British Isles accents. Pocket TTS can clone from either. Gradium's `phonon-31m-en` (Apache-2.0, released 2026-10-19) has 37 English voices. Neither has emotion control. Every model found with real emotion control needs a GPU or a conditional license.
8. Several permissive-looking tools hide non-commercial parts. `ctc-forced-aligner` is BSD-2 but its default weights are CC-BY-NC. NISQA code is MIT but its weights are CC-BY-NC-SA. The `ears/` and `expresso/` folders in `kyutai/tts-voices` are non-commercial. OmniVoice, Fish S2 Pro, Fish Drama-3, Higgs TTS 3, Breeze TTS 2, CrisperWhisper 2.0 and `phonon-40m` are also out.
9. Moonshine, Kokoro, Silero, Smart Turn, Pocket TTS and Whisper already have Pipecat services, which lowers integration cost. Phonon's Pipecat service (`GradiumTTSService`) is a WebSocket client whose local use is unverified.

## 1. Speech-to-speech and realtime voice

### 1.1 LFM3-Audio-2B

| Field | Value |
| --- | --- |
| Repos | `LiquidAI/LFM3-Audio-2B`, `LiquidAI/LFM3-Audio-2B-GGUF`, `LiquidAI/LFM3-Audio-2B-ONNX` (repo 19.74 GB) |
| Released | 2026-10-27 |
| Parameters | 2B, LFM3 backbone, causal audio encoder, LiquidCodec |
| GGUF main | Q4_0 1.05 GB, Q4_K_M 1.10 GB, Q8_0 1.85 GB, F16 3.47 GB |
| GGUF mmproj | Q4_0 0.16 GB, Q4_K_M 0.17 GB, Q8_0 0.30 GB, F16 0.57 GB |
| Audio in / out | 16 kHz PCM16 mono in, 24 kHz PCM16 mono out |
| Context | 32K tokens, input plus output |
| llama.cpp | b11960 (v0.8.0) or newer. Tool calls in realtime came with #31287. Use b12019 or later: mmproj files were requantized on 2026-10-29 (#31386) |
| License | LFM Open License v1.0 for code and weights |

Docs conflict: docs.liquid.ai lists LFM3-Audio as GGUF only, while the model card and Hub show ONNX and MLX repos too.

**Capabilities.** ASR in 18 languages, TTS in 19 locales including `en` and `en-gb`, translation, audio understanding, spoken tool calls. Voice chat in speech is English only. Text-only answers can reason with a `<think>` prefill. Live ASR is English only and works in chunks of about 0.96 s, which is too slow for barge-in detection.

**Modes and required prompts (vendor).**

| Mode | System prompt | Prefill |
| --- | --- | --- |
| ASR | `Perform ASR in English.` | none |
| Live ASR | `Perform live ASR.` | none |
| TTS | `Perform TTS in American English using a female voice.` | `<\|audio_start\|>` |
| Voice chat | `Respond with interleaved text and audio.` | `<\|mixed_start\|>` |
| Tools with speech | `Respond with function calls, and interleaved text and audio.\n\nList of tools:` | none, the model picks |
| Audio questions | `Answer questions about the provided audio clip. Base every answer only on what can be heard.` | none |

**Sampling.** Text uses temperature 0. Speech uses audio temperature 1.0, top_k 128, repetition penalty 1.1 over 4096 frames. These are stored in the GGUF. Liquid's published evals used audio temperature 0.4, top_k 0, penalty 1.2 over 64 frames, so scores may not match card defaults.

**Scores (vendor, model card and blog).**

| Benchmark | LFM3-Audio 2B | LFM3-Omni 2B | Gemma 4 E2B | Gemma 4 E4B | Qwen2.5-Omni 3B | Voxtral Mini 3B | MiMo-Audio 7B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Total params | 2B | 2.2B | 5B | 8B | 6B | 5B | 8B |
| English WER, lower better | 6.23 | 6.36 | 8.26 | 7.82 | 6.30 | 5.51 | 8.5 |
| VoiceBench | 73.13 | 64.73 | 63.28 | 72.90 | 66.37 | 65.78 | 77.3 |
| Tool calling, internal, audio in | 58.50 | 52.20 | 55.60 | 72.40 | 44.30 | 44.50 | 33.4 |
| MMAR audio understanding | 50.60 | 51.40 | 44.49 | 49.10 | 54.41 | 50.10 | 62.3 |
| English TTS UTMOSv2 | 3.83 | 3.80 | none | none | 3.42 | none | 2.8 |

Quantization barely moves these (PR author's evals on #31386, using `/v1/realtime`): FLEURS en-US WER stays between 4.31 and 4.57 for any Q4 or Q8 pairing, and SeedTTS UTMOSv2 stays between 3.79 and 3.85. The model was trained with QAT.

**Reported latency.** Time to first token "as low as 50 ms" (vendor, no hardware named). LiquidCodec decodes at about 87 times real time on one Apple M4 Max CPU thread (vendor). PR #31287 shows "first output 277 ms after the end of the input" for LFM3-Omni Q8_0 with a tool call, hardware not stated. Nothing is published for AMD audio.

**`/v1/realtime` protocol (source: `tools/server/server-realtime.cpp`, README).**

Connect to `ws://host:port/v1/realtime`. Default port is 9931 unless `--port` is set. The Hub card uses `llama serve` and the examples use `llama-server`; the Vulkan tarball contains both `llama` and `llama-server`. Sessions opened with `?intent=transcription` do live ASR on models with an `asr` template. Anything else is a conversation.

Client events accepted:

| Event | Behavior |
| --- | --- |
| `session.update` | `type`, `instructions`, `output_modalities`, `max_output_tokens`, `tools`, `tool_choice`. Type, instructions and tools are rejected after the first audio or message |
| `input_audio_buffer.append` | Base64 PCM16 mono at `audio.input.format.rate` |
| `input_audio_buffer.commit` | Ends the audio of the turn. Errors if the buffer is empty |
| `input_audio_buffer.clear` | Replies `cleared` but discards nothing, because the audio "already went into the conversation" |
| `conversation.item.create` | Only a user `input_text` message or a `function_call_output`. Assistant items are rejected |
| `response.create` | Ends the turn and commits pending audio. Accepts `output_modalities`, `max_output_tokens`, `tool_choice` |
| `response.cancel` | Stops generation, then `response.done` with status `cancelled` |

Any other event type returns `error: unknown event type`.

Server events: `session.created`, `session.updated`, `input_audio_buffer.committed`, `conversation.item.added`, `conversation.item.done`, `response.created`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_audio.delta` (base64 PCM16), `.done` variants, `response.function_call_arguments.delta` and `.done`, `response.done`, `error`.

`output_modalities`: omitted lets the model choose, `["text"]` bans the audio markers, `["audio"]` forces `<|audio_start|>`, `["text","audio"]` forces `<|mixed_start|>`.

**VAD and turn handling.** There is none on the server. `session_json()` always sends `turn_detection: null`, and no `speech_started` or `speech_stopped` events exist. The client decides when a turn ends and sends `response.create`. Audio prefills as it arrives, so the response can start right after the last chunk.

**Consequences for barge-in (source).**

- Audio appended during a response waits in the slot's op queue until the response ends. The slot only drains ops when `responding` is false. The client must send `response.cancel` at VAD onset.
- A cancelled response stays in context. Generation runs faster than playback, so the model remembers text the caller never heard. There is no truncate event.
- The brief's rule that context reflects only what was heard needs a workaround, because assistant items cannot be written. Two options: add a user `input_text` note next turn saying what was actually heard, or close the session and open a new one with the heard transcript in `instructions`. Neither is tested.
- `input_audio_buffer.clear` is a no-op, so a VAD false start or echo leak goes permanently into context. Gate mic audio with local VAD and echo handling before sending it. Pre-roll has to be handled client-side too.
- Conversation sessions emit no user transcript. Transcription events exist only in `transcription` sessions, so "heard text" for the call artifact must come from a separate STT.
- Interleaved `response.output_text.delta` gives the intended text for the artifact.
- If context fills, the response ends early. Decoding more then fails with `the conversation does not fit in the context`. How fast a call consumes 32K is undocumented.

**Concurrency (source).** Each session holds a slot, and the code comment says "several slots respond in turns on the same context". Run two server processes on separate ports for bot-to-bot use, each costing its own memory. Check the per-slot `n_ctx` in the log if `--parallel` is above 1.

**Open bug.** Issue #31436, open as of 2026-10-30: forcing `["audio"]` makes speech unintelligible, while `["text","audio"]` works (reported on Linux with CUDA). Use `["text","audio"]`.

**Tool calling.** Pass `tools` in `session.update` before any audio. Output goes through the chat template's parser and grammar with parallel calls on. Calls arrive as `response.function_call_arguments.*`. After `response.done`, send one `function_call_output` per call, then `response.create`. Native format is `<tool_call><function=name><parameter=x>...`. `tool_choice` accepts `auto`, `none`, `required`.

**Voices.** Two per locale, `female` and `male`, selectable through `--tts-voice` and `POST /tts` (`voice` field). `/v1/realtime` has no voice field, and no documented way to pick one in voice chat. No cloning (`--tts-speaker-file` is not used). A Liquid cookbook notebook fine-tunes a new consistent voice from LJSpeech, so a custom bot voice is possible but persona variety is not.

**License (source: LICENSE).** Based on Apache 2.0 with Section 5 added.

- Commercial use is licensed only while You or Your Legal Entity stay under the Threshold, defined as annual revenue of USD 10 million or more. Legal Entity includes everything under common control, at 50 percent or by contract.
- Applies to derivative works, including fine-tunes.
- Downstream customers are measured on their own revenue (user guide).
- Internal research, testing and evaluation need no commercial license (user guide). Using it in a business can count as commercial even with no direct revenue (user guide).
- Qualified 501(c)(3)-type nonprofits are exempt for non-commercial or research use.
- Section 11: the license terminates automatically on breach, and all copies must be deleted.
- No copyleft. You may keep and relicense modifications if your own use still complies.

Fine for this prototype. A move to a company at or above USD 10 million revenue needs a paid license from Liquid, which affects every `LFM3-*`, `d1-*` and `kani-tts-2-en` model. Adapters must stay swappable.

**Vulkan status.**

- The ops in `lfm3a.cpp` and `lfm3a-gen.cpp` are `SSM_CONV`, `ROPE`, `CONV_2D_DW`, `PAD`, `GET_ROWS` (all Vulkan supported in `docs/ops.md`) and `CONV_2D` (partial). The ISTFT runs on the host. mtmd logs unsupported ops at load.
- LFM3-1.7B on Vulkan on this chip is proven by Liquid's own table. The audio projector and realtime pipeline are not.
- Search found no public report of LFM3-Audio on Vulkan or AMD.
- Compare with `--no-mmproj-offload`, since the codec is fast on CPU.
- The cache holds a Q8 copy. Re-pull if it predates 2026-10-29.

### 1.2 LFM3-Omni-2B

`LiquidAI/LFM3-Omni-2B-GGUF`, 2026-10-27, Q8_0 1.85 GB plus mmproj 0.51 GB, same license. Same `/v1/realtime` path. Needs `--reasoning off`. Audio temperature 0.4, penalty 1.2. Lower VoiceBench (64.73) and tool score (52.20) than LFM3-Audio, slightly higher MMAR (51.40). Video and images add nothing for this project. No reason to prefer it for the bot.

### 1.3 Other speech-to-speech candidates

| Model and repo | Date | Size | Non-CUDA path | Reported latency | Code / weights | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Nemotron Labs VoiceChat, `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` | 2026-07-29 | repo 44.4 GB | None. NeMo-Speech.cpp `cuda-s2s` only, 16 GB CUDA minimum | ~450 ms (vendor) | Apache / OpenMDW-1.1 | Best open full-duplex with tools, CUDA only. Pipecat's Spark notes say it can self-talk and repeat canned replies |
| Moshi, `kyutai/moshiko-pytorch-bf16`, `kyutai/moshika-rag-pytorch-bf16` | 2024-09-11, 2026-03-19 | repo 15.8 GB each | PyTorch on ROCm unverified. Rust build not checked for AMD | 200 ms on L4 GPU (vendor) | Apache-2.0 / CC-BY-4.0 | No tool calling. Out for the bot |
| PersonaPlex, `nvidia/personaplex-7b-v1` | 2025-12-31 | repo 17.1 GB, gated | Unverified, card gated | Not read | NVIDIA Open Model License | No evidence of tools or a non-CUDA path |
| `kyutai/personaplex-rl-seamless` | 2026-06-02 | repo 17.1 GB, gated | Unverified | Not read | Terms not read | Same |
| `kyutai/moshika-rl-seamless` | 2026-06-02 | | | | CC-BY-NC-4.0 | Ruled out |
| Kyutai Unmute, `kyutai-labs/unmute` | Repo | Stack | None: CUDA, 16 GB VRAM | 450 to 750 ms TTS (vendor) | MIT code | A cascade stack, not a model. Out |
| Realtime-Venus, `inclusionAI/Realtime-Venus` | 2026-09-16 | repo 640 GB | None, examples need CUDA | Not stated | Apache-2.0 | 9B full-duplex with delegation. Out |
| MiniCPM-o 4.5, `openbmb/MiniCPM-o-4_5` | 2026-02-03 | repo 20 GB | Card mentions llama.cpp, speech output via it unverified | Not stated | Apache-2.0 tag | Full-duplex, 9B. Test only if PyTorch on ROCm works |
| Ming-3.0-audio-tiny, `inclusionAI/Ming-3.0-audio-tiny` | 2026-09-03 | repo 20.8 GB | None found | Not stated | MIT | CUDA examples. Out |
| Covo-Audio-Chat, `tencent/Covo-Audio-Chat` | 2026-03-16 | repo 18.7 GB | None found | Not stated | Custom Covo-Audio terms, not fully read | Out |
| Apriel Voice 27B, `ServiceNow-AI/Apriel-Voice-27B-preview` | 2026-10-13 | repo 54.5 GB | No GGUF found | 1.91 s mean, 2-stage (vendor) | MIT | Text output only. EVA-A 51.27, τ-Voice 65.51. Needs vLLM |
| Raon SpeechChat, `KRAFTON/Raon-SpeechChat-9B` | 2026-04-01 | | | | CC-BY-NC-4.0 | Ruled out |
| Qwen Omni, `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` | 2026-04-13 | Q4_K_M 18.56 GB plus mmproj 1.33 GB | llama.cpp audio input | N/A | Apache-2.0 | Text output only in llama.cpp. See section 6 |
| Gemma 4 | | | llama.cpp audio input | | Apache-2.0 | Text output only, 30 s audio limit. See section 6 |
| DuplexOmni, arXiv 2606.09186 | 2026-06 | | | | | Paper says weights will be released. No repo found. Unverified |
| Block Dessa | 2026-10 | | | | | Research preview. No weights, parameter count or benchmarks. Claims of open release are unconfirmed |

Moshi and PersonaPlex have no tool calling evidence, so they could only serve as a caller voice, where a fixed fact sheet and beats would be hard to enforce.

## 2. STT for live use

### 2.1 Runtimes without CUDA

| Runtime | Version | License | GPU on this box | Notes |
| --- | --- | --- | --- | --- |
| NeMo-Speech.cpp, `NVIDIA/NeMo-Speech.cpp` | v0.3.0, 2026-10-30 | Apache-2.0 | Vulkan tarball, `--device vulkan:0` | HTTP `/v1/audio/transcriptions`, WS `/v1/audio/transcriptions/realtime`. Session accepts `sample_rate` 8000 to 96000, `verbatim`, `word_timestamps`, `endpointing_ms`. `words[]` has `start`, `end`, `confidence` |
| llama.cpp | b12019 | MIT | Vulkan tarball | `/v1/audio/transcriptions`, `/v1/realtime?intent=transcription` |
| transcribe.cpp, `handy-computer/transcribe.cpp` | v0.4.1, 2026-10-22 | MIT | `linux-x86_64-cpu-vulkan` tarball | 16 families, Python bindings. GGUFs under `handy-computer/` |
| CrispASR, `CrispStrobe/CrispASR` | Active | MIT | `-vulkan` tarball, no CPU fallback in that build | 119 backends, HTTP server. Community project |
| whisper.cpp | v1.9.5, 2026-10-06 | MIT | Docker `ghcr.io/ggml-org/whisper.cpp:main-vulkan` with `--device /dev/dri`. Source build needs `libvulkan-dev` and `glslc`, so no sudo means Docker | 30 s window |
| sherpa-onnx | v2.0.1, 2026-10-26 | Apache-2.0 | CPU | No Vulkan support seen in README |
| moonshine-voice | Active | MIT | CPU only, by design | Python, Pipecat `MoonshineSTTService` |

ONNX Runtime v1.31.1 has no Vulkan provider. Its tree has `migraphx`, `webgpu`, `vitisai`. MIGraphX and the NPU path are unverified here. Treat ONNX models as CPU.

### 2.2 Models

| Model and repo | Date | Size | Runtime | Reported speed | WER | Word times | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parakeet TDT 0.6B v3, `nvidia/parakeet-tdt-0.6b-v3` | 2025-08-04 | Q8_0 0.71 GB | NeMo-Speech.cpp, transcribe.cpp, sherpa-onnx | parakeet.cpp 45x, sherpa-onnx 42x on 8 Zen 5 cores (third party) | 4.96 average per Phonon table | Yes, with confidence in NeMo-Speech.cpp | Apache / CC-BY-4.0 | Batch model. Needs VAD segments |
| Nemotron Speech Streaming EN 0.6B, `nvidia/nemotron-speech-streaming-en-0.6b` | 2025-12-17 | Q8_0 0.70 GB | NeMo-Speech.cpp, transcribe.cpp | 27 ms per 160 ms chunk, 6x on i7-11700K CPU (vendor) | 6.93 at 1.12 s, 7.07 at 0.56 s, 7.67 at 0.16 s, 8.43 at 0.08 s | Yes | Apache / NVIDIA Open Model License | AMI WER is 18.29 at 0.08 s |
| Nemotron 3.5 ASR Streaming, `nvidia/nemotron-3.5-asr-streaming-0.6b` | 2026-05-15 | Q8_0 in repo | Same | Same class | 7.96 average, third-party table | Yes | Apache / OpenMDW-1.1 | Card recommends the English model for English only |
| Parakeet Ultra, `moondream/parakeet-ultra` | 2026-09-22 | 1.26 GB | transcribe.cpp `handy-computer/parakeet-ultra-gguf` | Not measured on CPU | 5.80 vs 6.26 on their 7-set scoring, noise 5.82 vs 6.72 (vendor) | Through runtime | CC-BY-4.0 | New retrain of v3 |
| Parakeet Redux, `moondream/parakeet-redux` | 2026-09-18 | 178 MB | Photon, `pip install moondream` | 113x on 8 Zen 5 cores (vendor) | 6.55, noise 9.04 vs 6.72 (vendor) | Yes | CC-BY-4.0 weights, Photon license unstated on PyPI | Worse in noise. Runtime license unknown |
| Phonon-2, `FermionResearch/Phonon-2` | 2026-09-28 | 164 MB | `ghcr.io/fermionresearch/phonon-cpu:2.0.7`, transcribe.cpp | 143x on 8 Zen 5 cores (vendor) | 5.21 | Start and end with `--json`. Confidence not documented | Apache / CC-BY-4.0 | English. New |
| Underdog-Bark-3.0, `ConwayResearch/Underdog-Bark-3.0` | 2026-10-15 | 163 MB | Mac first. Generic path expands to FP32 PyTorch | 41x on M2 (vendor) | 4.94 (vendor, not on leaderboard) | Not checked | CC-BY-4.0 | No efficient x86 runtime |
| Moonshine Medium Streaming, `moonshine-ai/moonshine-streaming` | 2025-12-25 | 245M params | moonshine-voice | 269 ms on "Linux x86", 59 ms on MacBook Pro (vendor) | 6.65 | Yes, with confidence | MIT | Fixed 8-bit `.ort`, CPU only |
| Voxtral Realtime, `mistralai/Voxtral-Mini-4B-Realtime-2602` | 2026-01-21 | `ggml-org/...-GGUF` Q4_K_M 2.55 GB plus mmproj 1.06 GB | llama.cpp `/v1/realtime?intent=transcription` | 480 ms delay recommended (vendor). CPU and Vulkan speed unpublished | 6.12 average, third-party table | No word times in llama.cpp events | Apache-2.0 | Largest option here. Events carry text only |
| Qwen3-ASR 0.6B, `ggml-org/Qwen3-ASR-0.6B-GGUF` | 2026-01-28 | Q8_0 0.80 GB plus 0.21 GB, cached | llama.cpp, transcribe.cpp | Unpublished on CPU | 1.7B is 4.94 on 9 sets (via NVIDIA card) | Separate aligner | Apache-2.0 | Streaming is vLLM only, and no timestamps when streaming |
| Kyutai STT, `kyutai/stt-1b-en_fr`, `kyutai/stt-2.6b-en` | 2025-06 | repo 2.4, 5.6 GB | `moshi-server` | L40S 64 streams at 3x (vendor). CPU unpublished | Not read | Yes | Apache / CC-BY-4.0 | GPU oriented. Skip |
| Whisper large-v3-turbo | 2024-10-01 | 1.62 GB, q5_0 0.57 GB | whisper.cpp Vulkan | Not measured | 6.58, third-party table | `-ml 1`, experimental | MIT | 30 s window, poor partials |
| distil-large-v3 with faster-whisper | Local | Local | CPU | Working in `~/projects/whisper-bot/services` (local) | | | MIT | Known fallback |
| VibeVoice-ASR-BitNet, `microsoft/VibeVoice-ASR-BitNet` | 2026-07-24 | 1.58 GB | `microsoft/VibeASR.cpp` | RTF 1.98 at low threads to 0.42 at high on EPYC 7V13 (vendor) | Not read | Who-what-when | MIT | Post-call fit, not live |
| Audio8-ASR-Infinite, `Edge0/Audio8-ASR-Infinite` | 2026-09-21 | repo 8.2 GB | Adapted vLLM | 240 to 560 ms delay | Not read | Not read | Apache-2.0 | Needs vLLM. Skip |
| Whistle, `Cactus-Compute/whistle` | 2026-09-30 | 16.9 MB | Needle C++ engine | Not measured | Not comparable | Yes, with probability | Apache-2.0 | Tiny model, WER not compared |

Segmented STT may beat streaming here. At 45x, a 3 s utterance takes about 70 ms, and Parakeet TDT's WER is 4.96 against 7.67 for streaming at 0.16 s. Barge-in comes from VAD, not from partials, so streaming's advantage is smaller than it looks. Test both, since one binary runs both. Zen 5 numbers above came from an EPYC 9575F. Strix Halo is also Zen 5 with AVX-512, but clocks and memory differ.

## 3. Independent post-call ASR

The brief requires independence and no domain hints. Use a different model family from the live STT, and do not use keyword biasing. All models below take 16 kHz, so 8 kHz mu-law gets upsampled and content above 4 kHz is simply missing. NeMo-Speech.cpp accepts 8 kHz input and resamples.

| Model and repo | Size | Telephone evidence | Word confidence | Runtime | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- | --- |
| Granite Speech 5.0 2B, `ibm-granite/granite-speech-5.0-2b-GGUF` | Q8_0 1.96 GB plus mmproj 1.16 GB | Training lists Switchboard 260, CallHome 18, Fisher 2000 (units as in card, presumably hours) | Not documented | llama.cpp `/v1/audio/transcriptions`, Q8_0 recommended (vendor) | Apache-2.0 | Word timestamps not documented. Do not pass keywords |
| Whisper large-v3, `ggerganov/whisper.cpp` `ggml-large-v3.bin` | 3.10 GB, q5_0 1.08 GB | None published | Token `p` in `-ojf` JSON | whisper.cpp Vulkan in Docker | MIT / Apache-2.0 tag on `openai/whisper-large-v3` | Language prior can correct mispronunciations. Hallucination reported on poor TTS (Inflect card) |
| Nemotron 3.5 Transcribe, `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` | Community GGUF only, repo 67.5 GB | CALLHOME CH109 cpWER 11.50, the only telephone number found. Fisher and Switchboard in training | 0 for SALM models in NeMo-Speech.cpp | NeMo-Speech.cpp standard path runs on CPU and Vulkan, with no batching or speculation | Apache / OpenMDW-1.1 | Best published: Open ASR 4.71, long-form 10.98. Has a `verbatim, including fillers and repetitions` prompt, which suits pronunciation checks. 34B total, 4.3B active. Heavy, conversion needed |
| Parakeet TDT 0.6B v3 | Q8_0 0.71 GB | Fisher in training | Yes in NeMo-Speech.cpp | Vulkan tarball | Apache / CC-BY-4.0 | Not independent if live STT is Parakeet or Nemotron |
| Qwen3-ASR 1.7B plus `Qwen/Qwen3-ForcedAligner-0.6B` | `ggml-org/Qwen3-ASR-1.7B-GGUF`, aligner 1.84 GB | None | No | llama.cpp, PyTorch aligner | Apache-2.0 | Aligner needs PyTorch |
| Cohere Transcribe, `CohereLabs/cohere-transcribe-03-2026` | repo 4.13 GB, gated | None | No, and no timestamps | transcribe.cpp GGUF with Vulkan | Apache-2.0 | 5.42 in March 2026. No word output |
| Granite 5.0 TurboCTC, `ibm-granite/granite-speech-5.0-470m-turboctc` | `handy-computer/granite-speech-5.0-470m-turboctc-gguf` | 240 units of written-form numbers, codes, URLs at 16 and 8 kHz | Derivable from CTC posteriors, not exposed by transcribe.cpp | transformers, transcribe.cpp | Apache-2.0 | Avoid sibling `-nc`. transcribe.cpp doc says English only, possibly stale after the 2026-10-21 update |
| MOSS-Transcribe-Diarize, `OpenMOSS-Team/MOSS-Transcribe-Diarize` | repo 1.83 GB | "calls" named as a use case | No | transcribe.cpp, SGLang-Omni | Apache-2.0 | Segment timestamps. Useful for uploaded mono recordings |
| Nemotron 3 Diarization, `nvidia/Nemotron-3-Diarization` | repo 0.71 GB | | | NeMo-Speech.cpp | OpenMDW-1.1 | Also only for uploaded mono recordings, since system calls are per-channel |

Ruled out: `nyralabs/CrisperWhisper2.0_large`, the closest match for verbatim transcription, under a non-commercial research license. `netease-youdao/Confucius4-R2T2` uses a custom NetEase agreement I did not read.

Note: no model card publishes WER on simulated 8 kHz mu-law audio with packet loss. Measure it with this project's phone line.

## 4. TTS

### 4.1 Fast enough for real time without CUDA

| Model and repo | Date | Size | Voices | Runtime | Reported speed | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phonon 31m, `gradium/phonon-31m-en` | 2026-10-19 | q8 64.5 MB | 102 total in `gradium/phonon-voices`, 37 English: 13 F, 12 M, 12 characters. Apache-2.0 | `ptts` Rust CLI, Python, WebSocket and OpenAI-compatible servers, GGUF weights in its own runtime. 24 kHz | 26x, 5 ms TTFB on 1 M5 core. 2.3x, 68 ms on 1 Pi 5 core (vendor). No x86 number | MIT or Apache / Apache-2.0 | 11 days old. Rate control only, no emotion. Cloning and voice design go through Gradium's studio, pricing not checked |
| Pocket TTS, `kyutai/pocket-tts` (gated, cloning), `kyutai/pocket-tts-without-voice-cloning` (cached) | 2025-12-29, updated 2026-10-21 | Small models 18.2M, 6.9M, 1.9M params | Small catalog plus clone from WAV, see 4.3 | Python, Rust binary with no Python, `pocket-tts serve`, llama.cpp `llama-tts`, Pipecat `PocketTTSService` | ~200 ms first chunk, ~6x on M4 Air CPU, 2 cores. Small models up to 25x on one thread (vendor) | MIT / CC-BY-4.0 | Cloning needs accepting Hub terms and `hf auth login`. No emotion control |
| Kokoro 82M, `hexgrad/Kokoro-82M`, `onnx-community/Kokoro-82M-v1.0-ONNX` | 2024-12-26 | 82M params | 54 total, 28 English: 20 US, 8 UK. Blending allowed | `kokoro-onnx`, sherpa-onnx, Pipecat `KokoroTTSService` | 7.3x, 776 ms TTFB on one M5 core in PyTorch (Gradium's comparison, competitor-reported) | Apache / Apache-2.0 | Generates a sentence at a time. Uses espeak-ng. No emotion |
| Inflect Micro v2, `owensong/Inflect-Micro-v2` | 2026-06-25 | 37.5 MB | 1 fixed | PyTorch, ONNX | 6.28x on 4 threads, UTMOS22 4.395 (vendor) | Apache-2.0 | One voice |
| Inflect Nano v2, `owensong/Inflect-Nano-v2` | 2026-06-25 | Under 4M params | 1 fixed | Same | 10.72x on 4 threads (vendor) | Apache-2.0 | One voice |
| Soprano 1.1, `ekwek/Soprano-1.1-80M` | 2026-01-14 | 80M params | 1 fixed | llama.cpp `llama-tts` after conversion | Not read | Apache-2.0 | English, normalized input only |
| MagpieTTS, `nvidia/magpie_tts_multilingual_357m` | 2025-12-11 | Q8_0 0.33 GB | 5 English | NeMo-Speech.cpp, Vulkan tarball | 203 ms first audio, 2.7x CPU (vendor) | Apache / NVIDIA Open Model License | 22 kHz. Needs codec and frontend files |
| MOSS-TTS-Nano, `OpenMOSS-Team/MOSS-TTS-Nano-100M` | 2026-04-02 | 0.24 GB | Clone from WAV | PyTorch `infer.py`. 4-core CPU streaming (vendor) | Not quantified | Apache-2.0 | Clone is the main mode. 48 kHz stereo native |
| Supertonic 3, `Supertone/supertonic-3` | 2026-05-06 | 99M params | 10 styles, 5 F and 5 M | ONNX Runtime | Chart only, no number read | MIT code / OpenRAIL with use limits | GitHub repo archived: "Development and support have ended" |
| Piper, `rhasspy/piper-voices` | Rolling | Per voice | Many, per-voice licenses | `OHF-Voice/piper1-gpl`, Pipecat service | Not read | GPL-3.0 runtime / MIT tag, per-voice varies | GPL runtime. Older-sounding |
| LFM3-Audio `/tts` | 2026-10-27 | Shared with S2S | 2 | llama.cpp | See 1.1 | lfm1.0 | 2 voices |

### 4.2 Quality or control, likely too slow live on this GPU

| Model and repo | Date | Size | Control | Non-CUDA path | Reported speed | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-TTS 1.7B Base, CustomVoice, VoiceDesign | 2026-01-21 | repo ~4.5 GB. `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` Q4_K_M 1.04 GB plus mmproj 0.45 GB | Clone, 9 timbres, natural-language design and instructions | ROCm 7.2, llama.cpp `llama-tts` (a Vulkan crash was fixed in #28253) | ~0.8x on ROCm here (local), 97 ms claimed elsewhere (vendor) | Apache-2.0 | Below real time on 1.7B here. VoiceDesign is 1.7B only |
| Qwen3-TTS 0.6B Base, CustomVoice | 2026-01-21 | repo ~2.5 GB, cached | Clone, 9 timbres | ROCm 7.2. No official GGUF | Expected above real time, not measured (local note) | Apache-2.0 | `Qwen/` 0.6B GGUF absent from ggml-org |
| VoxCPM2, `openbmb/VoxCPM2` | 2026-04-03 | repo 5 GB | Design from description, age, emotion, pace | CUDA documented. ROCm unverified | RTF ~0.3 on RTX 4090, ~0.13 with Nano-VLLM (vendor) | Apache-2.0 | Use offline to create reference voices |
| Chatterbox Turbo, `ResembleAI/chatterbox-turbo`, `ResembleAI/chatterbox-turbo-ONNX` | 2025-12-02 | repo 4 GB, ONNX 7.4 GB | Clone, `[laugh]`-style tags | ONNX CPU, PyTorch ROCm unverified | Not published for CPU | MIT | All output carries the Perth watermark |
| Kitten TTS 2, `KittenML/kitten-tts-2` | 2026-09-30 | 947 MiB, 1.7B | 115 voices, `[emotion]` tags, emphasis, accents through voices such as `RussianBabushka` | `KittenML/kitten-tts-2-cpp` llama.cpp fork, CPU | Not published in card | Stellon Labs Community License | Commercial grant ends at USD 1M revenue or USD 1M cumulative funding. Must register for commercial use. Tighter than lfm1.0 |
| Ming-3.0-tts-0.5B, `inclusionAI/Ming-3.0-tts-0.5B` | 2026-09-22 | repo 3 GB | Clone, design, emotion, speed, pitch | CUDA tested only | Not read | MIT | Needs GitHub repo, not standalone |
| AuK, `tencent/AuK` | 2026-08-18 | repo 6.8 GB | Instruction TTS, emotion editing | CUDA expected, unverified | Not read | MIT | 1.5B foundation model |
| MOSS-TTS-Realtime, `OpenMOSS-Team/MOSS-TTS-Realtime` | 2026-02-09 | repo 4.7 GB | Clone, dialogue context | CUDA expected, unverified | Not read | Apache-2.0 | 1.7B |
| Audio8-TTS Realistic 0.6B, `Edge0/Audio8-TTS-Realistic-0.6b` | 2026-10-14 | repo 2.6 GB | Clone | CUDA recommended | Not read | Apache-2.0 | Unmeasured here |
| Gepard 1.0, `nineninesix/gepard-1.0` | 2026-06-22 | repo 1.1 GB | Clone | vLLM | ~25x, ~50 ms TTFA on RTX 5090 (vendor) | Apache-2.0 | GPU focused |
| Maya 2 Native, `maya-research/maya2-native` | 2026-10-26 | 5 GB | 20 voices, clone | CUDA expected, unverified | Not read | Apache-2.0 | Indian language focus |
| ZONOS2, `Zyphra/ZONOS2` | 2026-06-11 | repo 15.4 GB | Clone, expression | None: card requires CUDA | Not read | Apache-2.0 | Out |
| Orpheus, `canopylabs/orpheus-3b-0.1-ft` | 2025-03-17 | Gated, repo 56.7 GB | Tags | Community GGUF route unverified | Not read | Apache-2.0 | Older. Not worth testing first |
| CSM, `sesame/csm-1b` | 2025-03-06 | Gated, repo 19.6 GB | Context | Transformers | Not read | Apache-2.0 | Not realistic on CPU, unverified |
| Dia2, `nari-labs/Dia2-2B` | 2025-11-15 | repo 7.7 GB | Dialogue tags | CUDA expected, unverified | Not read | Apache-2.0 | Dialogue focus |
| Kani TTS 2, `nineninesix/kani-tts-2-en` | 2026-02-12 | repo 2.2 GB | | | | lfm1.0 | Same revenue limit as LFM3 |

### 4.3 Answers to the persona questions

**Which support many distinct voices without NC licenses?**

- Pocket TTS plus `kyutai/tts-voices`. Pocket weights are CC-BY-4.0. Folder terms come from that repo's README:
  - `voice-donations/`: CC0, 228 verified voices. Age and accent labels per voice are unverified, so someone would need to tag them.
  - `vctk/`: CC-BY-4.0. VCTK's own speaker metadata gives age, gender and accent, which suits persona selection.
  - `cml-tts/fr/`: CC-BY-4.0, French.
  - `ears/` and `expresso/`: CC-BY-NC-4.0, non-commercial. Exclude them.
  - `unmute-prod-website/`, `alba-mackenna/`, `voice-zero/`: mixed or unclear, per file. Exclude them unless checked. Pocket's default `alba` voice comes from `alba-mackenna`.
- `gradium/phonon-voices`: 37 English including 12 character voices, Apache-2.0.
- Kokoro: 28 English, Apache-2.0, plus blends.
- Qwen3-TTS VoiceDesign or VoxCPM2: unlimited offline voices from a text description, both Apache-2.0. Generate a reference clip per persona once, then clone it with Pocket TTS live. This gives explicit control of age and accent. It is an untested combination.
- Kitten TTS 2: 115 voices, but the license is limited, not NC.

**Which run on CPU, ONNX or Vulkan fast enough for real time?** With two sides and live STT sharing 16 cores, the likely set is Phonon 31m, Pocket TTS, Kokoro, MagpieTTS and Inflect. x86 numbers are missing for most, so measure them. Qwen3-TTS 1.7B is below real time here (local).

**Emotion control.** No real-time CPU model above has it. Delivery in live calls must come from voice choice, text and rate. Kitten, Qwen3-TTS, VoxCPM2, Chatterbox, Ming and AuK do have it, at a cost in speed or license. Turn-based bot-to-bot runs can use the slower models, since those runs do not need real time.

## 5. LLMs for the voice agent and the judge

### 5.1 Measured on Radeon 8060S with Vulkan (vendor)

Liquid's figures: Windows, llama.cpp, Q4_0, 32K context, batch 1, 100 output tokens, GPU offload. Other models used build b11744. Memory adds host and GPU peaks.

| Model | Prefill tok/s | Decode tok/s | Memory GB |
| --- | ---: | ---: | ---: |
| LFM3-1.7B | 3,771 | 129.6 | 2.75 |
| MiniCPM5-2B | 1,033 | 70.1 | 3.84 |
| Qwen3.5-4B | 1,355 | 51.2 | 6.00 |
| Qwen3.5-9B | 1,001 | 34.0 | 8.41 |
| Gemma 4 E4B | 337 | 37.6 | 9.52 |

Gemma's Vulkan prefill is an anomaly: the same chart shows 3,191 tok/s on Apple M5 Max. Check whether it persists on a current build. A 2,000 token system prompt at 337 tok/s takes about 6 s uncached. llama.cpp prompt caching should limit this to the first turn.

### 5.2 Candidates

| Model and repo | Date | GGUF | Tool scores (vendor) | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- |
| LFM3-1.7B, `LiquidAI/LFM3-1.7B-GGUF` | 2026-10-23 | QAD-Q4_0 1.05, Q4_K_M 1.10, Q8_0 1.85 GB | BFCL V4 71.39, ToolSandbox 80.36, MCP-Atlas 32.57, τ³-bench 68.01 | lfm1.0 | Reasoning on by default, so disable it. Not for complex multi-step reasoning (vendor). Use b12005 or later: `json_schema` with thinking off crashed until #31328 |
| Qwen3.5-4B, `unsloth/Qwen3.5-4B-GGUF` | 2026-02-27 | Q4_K_M 2.74 GB, cached in safetensors | BFCL 55.12, ToolSandbox 75.40, MCP-Atlas 41.37, τ³ 65.98 | Apache-2.0 | Reasoning off for latency |
| Qwen3.5-9B, `unsloth/Qwen3.5-9B-GGUF` | 2026-02-27 | Q4_K_M 5.68 GB | BFCL 62.83, ToolSandbox 76.41, MCP-Atlas 42.67, τ³ 67.11 | Apache-2.0 | 34 tok/s decode |
| Nemotron 3.5 Bolt E4B 4GB, `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` | 2026-10-05 | Q4_K_M 6.18 GB | ToolSandbox 74.13, RelyToolBench 69.29 at Q4 | OpenMDW-1.1 over Gemma Apache-2.0 | Gemma E4B post-train aimed at voice agents. Same Gemma graph, so the Vulkan prefill concern probably applies, unverified. llama.cpp recipe marked pending |
| Gemma 4 E2B, E4B, `ggml-org/gemma-4-E4B-it-GGUF` | 2026-03-02 | E4B Q4_0 4.59 GB, Q8_0 8.03 GB | E4B BFCL 48.25, ToolSandbox 69.27, MCP-Atlas 6.19, τ³ 33.53 | Apache-2.0 | Weak in Liquid's table, stronger in NVIDIA's |
| d1-2B, `LiquidAI/d1-2B-GGUF` | 2026-10-24 | | Decision model, not chat | lfm1.0 | Routing only |

These comparisons were chosen by the vendors. Liquid's table favors Liquid, and NVIDIA's favors Bolt.

Voice setting: temperature 0.1, top_k 50, repeat penalty 1.1 for LFM3-1.7B (vendor). Check that a tool-call turn does not leak a thinking block into speech.

### 5.3 Judge

Needs reasoning and structured output. No voice latency constraint.

| Model and repo | Date | Size | Code / weights | Notes |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B, `unsloth/Qwen3.8-27B-GGUF` | 2026-08-05 | UD-Q4_K_M 16.5 GB, UD-Q6_K 22.0 GB | Apache-2.0 | Newest of the set. Also on the user's Ollama, which must not be modified. Run a separate llama-server for isolation |
| Qwen3.6-35B-A3B, `unsloth/Qwen3.6-35B-A3B-GGUF` | 2026-04-15 | Safetensors cached, GGUF not sized | Apache-2.0 | Cached copy helps only via ROCm/vLLM unless a GGUF is also pulled |
| Gemma 4 26B-A4B, `ggml-org/gemma-4-26B-A4B-it-GGUF` | 2026-03-11 | Not sized | Apache-2.0 | Vulkan prefill worth checking |
| gpt-oss-20b, `ggml-org/gpt-oss-20b-GGUF` | 2025-08-02 | repo 14.75 GB | Apache-2.0 | Harmony format. Check `response_format` behavior |

Use llama.cpp `response_format` with `json_schema`. Speeds on this machine are unknown. The benchmarking agent should supply them.

## 6. Audio-understanding models

No model publishes accuracy on telephone delivery, tone or mispronunciation in support calls. The closest public number is MOSS-Audio's speech captioning table, scored by its authors, apparently out of 5. Advisory use, as the brief says, is the right bar.

| Model and repo | Date | Size | Non-CUDA path | Demonstrated | Code / weights | Catch |
| --- | --- | --- | --- | --- | --- | --- |
| LFM3-Audio-2B | 2026-10-27 | Q8_0 1.85 plus 0.30 GB | llama.cpp, Vulkan unverified | MMAR 50.6, VoiceBench 73.13 | lfm1.0 | Same model as candidate bot. Card names speech, sounds, events, not tone |
| LFM3-Omni-2B | 2026-10-27 | Q8_0 1.85 plus 0.51 GB | Same | MMAR 51.4, VoiceBench 64.73 | lfm1.0 | No gain |
| d1-Omni-2B, `LiquidAI/d1-Omni-2B-GGUF` | 2026-10-24 | Q8_0 1.85 plus 0.33 GB | llama.cpp (#31275), request format not read | Calibrated answers in one pass. 12.4 ms for a 10 s clip on H100 (vendor). "Call and voice-message routing" | lfm1.0 | No transcript or chat. Might suit yes/no flags per clip. Experiment only |
| MOSS-Audio-4B-Instruct, `OpenMOSS-Team/MOSS-Audio-4B-Instruct` | 2026-04-10 | repo 10.5 GB. `cstr/MOSS-Audio-4B-Instruct-GGUF` F16 9.73 GB | CrispASR `--backend moss-audio`, third party. PyTorch on ROCm unverified | Accent 4.497, clarity 3.841, fluency 3.744, emotion 3.311, tone 3.282 of 5 (vendor) | Apache-2.0 | Best public match to the task. Its table lists Gemini-3.1-Pro with numbers identical to Qwen3-Omni Instruct, likely a copy error, so read comparisons with care |
| Qwen3-Omni 30B A3B, `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` | 2026-04-13 | Q4_K_M 18.56 plus 1.33 GB | llama.cpp | Tone 3.224, emotion 3.328, accent 4.356 (MOSS table) | Apache-2.0 | Large. Also Thinking and third-party Captioner variants |
| Gemma 4 E2B, E4B | 2026-03-02 | E4B Q4_0 4.59 GB plus mmproj 0.56 GB | llama.cpp | ASR, translation (vendor). E4B MMAR 49.1 (Liquid) | Apache-2.0 | 30 s audio limit. Liquid ASR 7.82 WER |
| Gemma 4 12B | 2026-05-23 | `unsloth/gemma-4-12B-it-qat-GGUF` | llama.cpp | Encoder-free audio | Apache-2.0 | #24138: incoherent output on a 12 minute file, closed as stale, not fixed |
| MiMo-Audio-7B-Instruct, `XiaomiMiMo/MiMo-Audio-7B-Instruct` | 2025-09-18 | repo 16 GB | llama.cpp has the projector, no model GGUF found, conversion needed | Highest MMAR 62.3 and VoiceBench 77.3 in Liquid's table | MIT | Tool score 33.4 irrelevant here |
| Nemotron 3 Nano Omni, `ggml-org/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-GGUF` | 2026-07-30 | Q4_K_M 22.42 plus mmproj 1.74 GB | llama.cpp | Not read | NVIDIA Open Model Agreement | Large |
| Qwen2.5-Omni 7B, `ggml-org/Qwen2.5-Omni-7B-GGUF` | 2025-05-26 | repo 32.2 GB | llama.cpp | Not read | Apache-2.0 upstream, GGUF repo tagged qwen-research | License tags disagree. The 3B is qwen-research, so avoid both until checked |
| Step-Audio-R1.1, `stepfun-ai/Step-Audio-R1.1` | 2026-01-14 | repo 67 GB | No GGUF found | Not read | Apache-2.0 | Skip |
| Muse Spark Open, `ggml-org/Muse-Spark-Open-816B-A42B-GGUF` | 2026-10-14 | Very large | llama.cpp | Not read | Not read | Far outside 40 GB budget |
| Audio Flamingo Next, `nvidia/audio-flamingo-next-hf` | 2026-04-05 | | | | NVIDIA OneWay Noncommercial | Ruled out |

The chunking rule from the brief applies here too. Send only the segment under question, and ask for a timestamped claim with a confidence. For Gemma, keep clips under 30 s.

## 7. VAD, turn detection, echo

| Tool and repo | Date | Size | Runtime | Reported | License | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Silero VAD, `snakers4/silero-vad` | v6.2.3, 2026-09-23 | repo 0.01 GB | ONNX, PyTorch, bundled in Pipecat and NeMo-Speech.cpp | Under 1 ms per 30 ms chunk on one thread (vendor) | MIT | Native 8 kHz and 16 kHz. Best option for the mu-law leg |
| Smart Turn v3.2, `pipecat-ai/smart-turn-v3` | Repo 2025-09, v3.2 current | int8 8 MB, FP32 32 MB | ONNX. Pipecat ships `smart-turn-v3.2-cpu.onnx`, `LocalSmartTurnAnalyzerV3` | 10 ms on good CPUs, ~65 ms on 1x cloud instance (vendor) | BSD-2-Clause | Whisper Tiny base, 16 kHz, up to 8 s. Runs after VAD reports silence. Telephone band accuracy unknown |
| FireRedVAD, `FireRedTeam/FireRedVAD` | 2026-02-11 | repo 0.01 GB | PyTorch scripts | F1 97.57 on FLEURS-VAD-102 (vendor) | Apache-2.0 | 16 kHz only |
| Parakeet Realtime EOU, `nvidia/parakeet_realtime_eou_120m-v1` | 2025-10-10 | 0.46 GB | NeMo | Not read | NVIDIA Open Model License | Runtime support in NeMo-Speech.cpp unverified. NeMo-Speech.cpp itself lists "endpointing" |
| TYGER, `nineninesix/tyger` | 2026-10-12 | 2.61 GB | Own code. Two aligned streams, user mic and agent playback | 252.5 ms median, 29.9 percent early cut-ins, 415 of 1,000 pauses false endpoint (vendor) | OpenMDW-1.1 | Interesting for echo design, but too many false cuts |
| LocalVQE v1.3, `LocalAI-io/LocalVQE` | 2026-04-17 | ~19 MB | GGML C++, build from `localai-org/LocalVQE` | 3.2 ms per 16 ms hop, 4 threads on Ryzen 9 7900 (vendor) | Apache-2.0 | AEC, noise suppression, dereverb. Server-side echo candidate. Needs far-end reference |
| Hush, `weya-ai/hush` | 2026-03-19 | 8 MB | ONNX, PyTorch needs `DeepFilterLib` | Under 1 ms per 10 ms (vendor) | Apache-2.0 | Background speaker suppression, 16 kHz |
| Pipecat `rnnoise_filter` | | | Pipecat | | Check | Simple baseline filter |

Ruled out or not useful:

- TEN VAD: Apache-2.0 plus conditions that forbid deployment competing with Agora.
- LiveKit `turn-detector`: usable only with LiveKit Agents.
- `neuphonic/neudecide`: a speech-to-tool-call model, not turn detection, despite the name.
- `desert-ant-labs/uhm`: filler detector. Useful idea, but source-available with a paid tier at scale.
- Pipecat `krisp_viva` filters and turn: proprietary.

Pipecat v1.15.0 added `ClassifierUserTurnStrategies`, which runs a text classifier on top of Smart Turn. It defaults to a remote API in the changelog example.

## 8. Audio quality and pronunciation tooling

### 8.1 Quality

| Tool | Reference needed | Input rate | Code / weights | Use here |
| --- | --- | --- | --- | --- |
| DNSMOS, `microsoft/DNS-Challenge` `DNSMOS/DNSMOS/*.onnx` | No | 16 kHz | Code MIT, repo otherwise CC-BY-4.0, no separate weight license stated | Fast ONNX CPU. SIG, BAK, OVRL per file. Trained for enhancement, so compare to baseline |
| UTMOSv2, `sarulab-speech/UTMOSv2` | No | 16 kHz | MIT code, HF repo tagged MIT, 4.09 GB | Good for TTS, used in LFM3 benchmarks. CPU speed unknown |
| SCOREQ, `alessandroragano/scoreq` | Optional | 16 kHz, resamples | Code MIT. Weights on Zenodo, license unverified | Own table shows strong correlation on simulated network distortion. Check weight terms first |
| PESQ, `ludlows/PESQ` | Yes | 8 kHz narrowband mode or 16 kHz | Wrapper MIT. ITU-T P.862 reference code has its own IP terms, unverified | Only metric here with native 8 kHz. Works for simulated calls, since clean sent audio exists |
| ViSQOL, `google/visqol` | Yes | Speech mode at 16 kHz | Apache-2.0 | Full reference. Bazel build |
| NISQA | No | | Code MIT, weights CC-BY-NC-SA-4.0 | Ruled out |

Clipping, loudness (`pyloudnorm`), silence ratio and SNR estimates need no model.

### 8.2 Pronunciation and alignment

| Tool | What | Code / weights | Catch |
| --- | --- | --- | --- |
| `Qwen/Qwen3-ForcedAligner-0.6B` | Word or character timestamps for text and speech, up to 5 min, 11 languages | Apache-2.0 | PyTorch. Word level, not phoneme |
| Montreal Forced Aligner plus `english_us_arpa` v3.0.0 | Phone-level alignment | MIT code, CC-BY-4.0 model | Needs conda, no sudo required. Trained on wideband speech, so 8 kHz accuracy is unverified |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | CTC phoneme recognition | Apache-2.0 | Scoring expected phonemes needs espeak-ng or a lexicon. `phonemizer` is GPL |
| `m-bain/whisperX` | Word alignment via wav2vec2 | BSD-2-Clause | Check each alignment model's weights. Pyannote parts have their own terms |
| `MahmoudAshraf97/ctc-forced-aligner` | CTC alignment | BSD-2-Clause code | Default weights `MahmoudAshraf/mms-300m-1130-forced-aligner` are CC-BY-NC-4.0. Use another acoustic model |
| `charsiu/en_w2v2_fc_10ms` | Frame-level phone alignment | MIT code, no license tag on weights | Unverified weight terms. Avoid until clear |
| NeMo-Speech.cpp `verbatim` and `words[].confidence` | ASR confidence per word | Per model | Set `verbatim: true`, which skips inverse text normalization, so digits stay as spoken |

Suggested first pipeline, not tested: compare intended critical tokens to independent ASR with no hints. Where they disagree or word confidence is low, run forced alignment on that span with the intended text and flag the clip. Add phoneme CTC scoring only if this misses real errors.

## Ruled out by license

| Repo | License |
| --- | --- |
| `k2-fsa/OmniVoice` | Code Apache-2.0, weights CC-BY-NC (card), and tag is blank on Hub |
| `fishaudio/s2-pro`, `fishaudio/drama-3` | Fish Audio Research License, commercial use needs a separate license |
| `bosonai/higgs-tts-3-4b` | Research and non-commercial |
| `BreezeBlue/Breeze-TTS-2` | Research and non-commercial |
| `gradium/phonon-40m-*` | CC-BY-NC-4.0 |
| `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | CC-BY-NC-SA-4.0 |
| `nyralabs/CrisperWhisper2.0_large` | Non-commercial research |
| `Edge0/Audio8-ASR-0.1B` | CC-BY-NC-4.0 |
| `KRAFTON/Raon-Speech-9B`, `KRAFTON/Raon-SpeechChat-9B`, `KRAFTON/A.X-K2-Raon-Speech-21B-A3B` | CC-BY-NC-4.0 |
| `kyutai/moshika-rl-seamless` | CC-BY-NC-4.0 |
| `Qwen/Qwen2.5-Omni-3B` | qwen-research |
| Moonshine legacy non-English Base and Tiny models | Moonshine Community License. English and all streaming models are MIT |
| `nvidia/audio-flamingo-next-hf` | NVIDIA OneWay Noncommercial |
| `MahmoudAshraf/mms-300m-1130-forced-aligner` | CC-BY-NC-4.0 |
| `kyutai/tts-voices/ears`, `kyutai/tts-voices/expresso` | CC-BY-NC-4.0 |
| NISQA weights | CC-BY-NC-SA-4.0 |
| `desert-ant-labs/voz`, `desert-ant-labs/uhm` | Source-available, commercial license at scale |

Conditional, not NC: `LFM3-*`, `d1-*`, `kani-tts-2-en` (USD 10M revenue), `KittenML/kitten-tts-2` (USD 1M revenue or funding, registration required), `Supertone/supertonic-3` (OpenRAIL use limits).

## Unverified items and open questions

- LFM3-Audio on Vulkan on any AMD GPU: no public report. This is the top risk.
- Context consumption per minute of LFM3 realtime audio.
- Whether two realtime sessions on one server work acceptably. Two processes is the safe plan.
- Whether `response.cancel` flush returns speech after the cancel, and how much.
- Live ASR on the Q8 LFM3-Audio GGUF. PR #31257 said a follow-up, and the server README says it works with an `asr` template. Not confirmed on this repo.
- x86 RTF for Phonon 31m, Pocket TTS small models, Kokoro and Kitten TTS 2.
- `words[].confidence` quality in NeMo-Speech.cpp for Parakeet TDT.
- NeMo-Speech.cpp v0.3.0 tarball layout and Vulkan behavior on RADV or AMDVLK. Its docs say "CPU, Vulkan" for standard paths, and benchmarks show only NVIDIA and CPU.
- Photon runtime license (`moondream` on PyPI shows none).
- Whether Silero, Smart Turn and Pipecat behave acceptably on upsampled 8 kHz audio.
- `llama-mtmd-cli` audio path with `<__media__>` for Gemma. The command is modeled on LFM3's and may need adjustment.
- Open ASR Leaderboard current rankings. The Space did not load, so figures come from model cards citing it, with different dates and set counts (7, 8 or 9 sets). Do not compare across tables to the second decimal.
- Phoronix review of ROCm vs Vulkan on Strix Halo. Search snippet only, page returned 403.
- Age and accent labels for Kyutai donated voices.
- Terms for `kyutai/tts-voices/alba-mackenna`, `unmute-prod-website` and `voice-zero`.
- PersonaPlex cards were gated. I did not read them.
- `Qwen/Qwen2.5-Omni-7B` shows `apache-2.0` as license name, while `ggml-org/Qwen2.5-Omni-7B-GGUF` shows `qwen-research`.
- DuplexOmni weight release. Whether Realtime-Venus relates to it was not established.
- LFM3-Audio ONNX repo: 19.74 GB, execution provider support not read.

## Sources

LFM3:

- <https://huggingface.co/LiquidAI/LFM3-Audio-2B>
- <https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF>
- <https://huggingface.co/LiquidAI/LFM3-Omni-2B>
- <https://huggingface.co/LiquidAI/LFM3-1.7B>
- <https://huggingface.co/LiquidAI/d1-Omni-2B>
- <https://www.liquid.ai/blog/lfm3>
- <https://docs.liquid.ai/lfm/help/model-license>
- <https://docs.liquid.ai/deployment/on-device/llama-cpp>
- <https://github.com/Liquid4All/cookbook>

llama.cpp:

- <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-realtime.cpp>
- <https://github.com/ggml-org/llama.cpp/blob/master/examples/realtime_client.py>
- <https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md>
- <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
- <https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md>
- <https://github.com/ggml-org/llama.cpp/pull/31257>
- <https://github.com/ggml-org/llama.cpp/pull/31269>
- <https://github.com/ggml-org/llama.cpp/pull/31287>
- <https://github.com/ggml-org/llama.cpp/pull/31386>
- <https://github.com/ggml-org/llama.cpp/issues/31436>
- <https://github.com/ggml-org/llama.cpp/issues/31328>
- <https://github.com/ggml-org/llama.cpp/issues/24138>
- <https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0>

Speech-to-speech:

- <https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B>
- <https://arxiv.org/abs/2609.21967>
- <https://arxiv.org/abs/2606.09186>
- <https://huggingface.co/pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark>
- <https://github.com/kyutai-labs/moshi>
- <https://github.com/kyutai-labs/unmute>
- <https://huggingface.co/inclusionAI/Realtime-Venus>
- <https://huggingface.co/openbmb/MiniCPM-o-4_5>
- <https://huggingface.co/ServiceNow-AI/Apriel-Voice-27B-preview>
- <https://huggingface.co/tencent/Covo-Audio-Chat>
- <https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny>

STT and ASR:

- <https://github.com/NVIDIA/NeMo-Speech.cpp>
- <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>
- <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B>
- <https://huggingface.co/moondream/parakeet-redux>
- <https://huggingface.co/moondream/parakeet-ultra>
- <https://huggingface.co/FermionResearch/Phonon-2>
- <https://huggingface.co/ConwayResearch/Underdog-Bark-3.0>
- <https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602>
- <https://huggingface.co/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF>
- <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- <https://huggingface.co/ibm-granite/granite-speech-5.0-2b>
- <https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc>
- <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>
- <https://huggingface.co/microsoft/VibeVoice-ASR-BitNet>
- <https://github.com/moonshine-ai/moonshine>
- <https://github.com/handy-computer/transcribe.cpp>
- <https://github.com/CrispStrobe/CrispASR>
- <https://github.com/ggml-org/whisper.cpp>
- <https://github.com/k2-fsa/sherpa-onnx>
- <https://github.com/kyutai-labs/delayed-streams-modeling>

TTS:

- <https://huggingface.co/gradium/phonon-31m-en>
- <https://huggingface.co/gradium/phonon-voices>
- <https://github.com/gradium-ai/xn-ptts>
- <https://github.com/kyutai-labs/pocket-tts>
- <https://huggingface.co/kyutai/tts-voices>
- <https://huggingface.co/hexgrad/Kokoro-82M>
- <https://huggingface.co/owensong/Inflect-Micro-v2>
- <https://huggingface.co/Supertone/supertonic-3>
- <https://github.com/supertone-inc/supertonic>
- <https://huggingface.co/KittenML/kitten-tts-2>
- <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign>
- <https://huggingface.co/openbmb/VoxCPM2>
- <https://huggingface.co/ResembleAI/chatterbox-turbo>
- <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M>
- <https://huggingface.co/k2-fsa/OmniVoice>
- `~/projects/cc-tts/README.md` for local Qwen3-TTS numbers

LLM and audio understanding:

- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB>
- <https://huggingface.co/Qwen/Qwen3.8-27B>
- <https://huggingface.co/google/gemma-4-E4B-it>
- <https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct>
- <https://huggingface.co/cstr/MOSS-Audio-4B-Instruct-GGUF>
- <https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct>

VAD, turn, quality, alignment:

- <https://github.com/snakers4/silero-vad>
- <https://github.com/pipecat-ai/smart-turn>
- <https://huggingface.co/pipecat-ai/smart-turn-v3>
- <https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/openai/realtime/llm.py>
- <https://huggingface.co/FireRedTeam/FireRedVAD>
- <https://huggingface.co/LocalAI-io/LocalVQE>
- <https://huggingface.co/weya-ai/hush>
- <https://huggingface.co/nineninesix/tyger>
- <https://huggingface.co/neuphonic/neudecide>
- <https://github.com/TEN-framework/ten-vad>
- <https://huggingface.co/livekit/turn-detector>
- <https://github.com/sarulab-speech/UTMOSv2>
- <https://github.com/microsoft/DNS-Challenge>
- <https://github.com/alessandroragano/scoreq>
- <https://github.com/gabrielmittag/NISQA>
- <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner>
- <https://github.com/MahmoudAshraf97/ctc-forced-aligner>

## Ranked shortlist per role

**1. Speech-to-speech**

1. `LiquidAI/LFM3-Audio-2B-GGUF` Q8_0 with `llama-server /v1/realtime`, custom adapter, local VAD, `["text","audio"]`.
2. Nothing else qualifies on this hardware. `LFM3-Omni-2B-GGUF` is a weaker spare.
3. Default bot stays a cascade until tests show LFM3-Audio tool calls and barge-in are good enough.

**2. Live STT**

1. `nvidia/parakeet-tdt-0.6b-v3` Q8_0 through NeMo-Speech.cpp, segmented by VAD. Best WER with word confidence.
2. `nvidia/nemotron-speech-streaming-en-0.6b` Q8_0, same binary, 0.16 s or 0.56 s chunks, if partials are needed.
3. `FermionResearch/Phonon-2` on CPU, if Vulkan disappoints.
4. `moonshine-ai/moonshine-streaming` Medium, for fastest integration through Pipecat.
5. `ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF`, if the others fail on telephone audio.

**3. Post-call ASR.** Must differ from live family.

1. `ggerganov/whisper.cpp` `ggml-large-v3.bin` on whisper.cpp Vulkan, for word probabilities.
2. `ibm-granite/granite-speech-5.0-2b-GGUF` Q8_0, for telephone corpora in training. Use as a second opinion.
3. `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` verbatim prompt, if a GGUF route works and the size is acceptable.
4. Parakeet TDT v3, if live STT ends up outside the Parakeet family.

**4a. TTS for the support bot**

1. `gradium/phonon-31m-en`, if quality on critical tokens holds.
2. `onnx-community/Kokoro-82M-v1.0-ONNX`, the known baseline with a Pipecat service.
3. `kyutai/pocket-tts` small English models.
4. `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` on ROCm, for turn-based quality runs.

**4b. TTS for caller personas**

1. Pocket TTS with `kyutai/tts-voices/voice-donations` (CC0) and `vctk/` (CC-BY-4.0). Do not use `ears/` or `expresso/`.
2. `gradium/phonon-voices` English set.
3. Offline design with `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, then clone into Pocket TTS.
4. Kokoro voices and blends.
5. `KittenML/kitten-tts-2` only if the license is accepted, for emotion tags.

**5a. Voice agent LLM**

1. `LiquidAI/LFM3-1.7B-GGUF` Q8_0.
2. `unsloth/Qwen3.5-4B-GGUF` Q4_K_M, the Apache fallback.
3. `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`, if prefill is acceptable.
4. `unsloth/Qwen3.5-9B-GGUF` Q4_K_M.

**5b. Judge**

1. `unsloth/Qwen3.8-27B-GGUF` UD-Q6_K.
2. `Qwen/Qwen3.6-35B-A3B` through a GGUF.
3. `ggml-org/gemma-4-26B-A4B-it-GGUF`.

**6. Audio listener**

1. `LiquidAI/LFM3-Audio-2B-GGUF`, already cached, cheapest to try.
2. `cstr/MOSS-Audio-4B-Instruct-GGUF` through CrispASR, best public tone and fluency numbers.
3. `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`.
4. `ggml-org/gemma-4-E4B-it-GGUF` for clips under 30 s.

**7. VAD and turn**

1. Silero VAD 6.2.3 plus Smart Turn v3.2, both already in Pipecat.
2. `localai-org/LocalVQE` for server-side echo cancellation if browser AEC is not enough.
3. FireRedVAD as a comparison.

**8. Quality and pronunciation**

1. DNSMOS plus plain signal checks.
2. UTMOSv2 for TTS channels.
3. PESQ narrowband in simulated calls, license pending.
4. Qwen3-ForcedAligner for spans, MFA with `english_us_arpa` for phones.

## Test these on the machine

None of these commands have been run. Paths assume the repo root. Do not touch Ollama. Use project ports.

Download budget: running everything below pulls about 50 GB, over the 40 GB limit. Skip or delete Voxtral (3.6 GB), Bolt (6.2 GB) and Gemma E4B (5.2 GB) until the rest is measured, which leaves about 35 GB, of which the judge is 22 GB.

### Setup

```bash
mkdir -p .runtime models
B=b12019
curl -L -o .runtime/llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/$B/llama-$B-bin-ubuntu-vulkan-x64.tar.gz
tar xzf .runtime/llama.tgz -C .runtime     # creates .runtime/llama-b12019/
export LD_LIBRARY_PATH=$PWD/.runtime/llama-$B
export PATH=$PWD/.runtime/llama-$B:$PATH

curl -L -o .runtime/nemo.tgz https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.3.0/nemo-speech-0.3.0-linux-x86_64-vulkan.tar.gz
mkdir -p .runtime/nemo && tar xzf .runtime/nemo.tgz -C .runtime/nemo
export PATH=$(dirname "$(find $PWD/.runtime/nemo -name nemo-speech -type f | head -1)"):$PATH

curl -L -o .runtime/realtime_client.py https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/realtime_client.py
```

### LFM3-Audio realtime

```bash
llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q8_0 --temp 0 -c 32768 --parallel 1 -ngl 99 --port 8091
# repeat with --no-mmproj-offload and compare

uv run --with 'openai[realtime]' --with numpy python .runtime/realtime_client.py \
  --url http://127.0.0.1:8091/v1 --output-modalities text,audio --out answer.wav question.wav
uv run --with 'openai[realtime]' --with numpy python .runtime/realtime_client.py \
  --url http://127.0.0.1:8091/v1 --tools --out tool.wav set_volume.wav
curl -s http://127.0.0.1:8091/tts -H 'Content-Type: application/json' \
  -d '{"input":"Your balance is forty two dollars.","voice":"male"}' -o male.wav
```

Check:

- Log lines about unsupported Vulkan ops, and whether the mmproj loads on Vulkan at all.
- The client's `first output N ms` over 20 turns, with the p95.
- Whether audio comes out faster than real time on all turns.
- Tool call accuracy on the Fernhill tool set, not `set_volume`.
- A 10 minute conversation: watch context use and whether it hits the limit.
- Send `response.cancel` mid-response, then read what the model says it told the caller.
- 8 kHz mu-law upsampled input vs clean input.

### LFM3-1.7B and other LLMs

```bash
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q8_0 --jinja --reasoning off --temp 0.1 --top-k 50 --repeat-penalty 1.1 -c 32768 -ngl 99 --port 8092
llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --jinja --reasoning off -c 32768 -ngl 99 --port 8092
llama-server -hf nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB:Q4_K_M --jinja -c 32768 -ngl 99 --port 8092
```

Measure TTFT with the real system prompt, cold and cached. Run 50 Fernhill tool scenarios. Compare ROCm llama.cpp only if Vulkan TTFT is poor. A Phoronix review in October 2026 reportedly found Vulkan higher in throughput and ROCm lower in latency on Strix Halo. The page returned 403, so this is from a search snippet only.

### STT

```bash
hf download nvidia/parakeet-tdt-0.6b-v3 parakeet-tdt-0.6b-v3.q8_0.gguf --local-dir models
hf download nvidia/nemotron-speech-streaming-en-0.6b nemotron-speech-streaming-en-0.6b.q8_0.gguf --local-dir models

nemo-speech transcribe clip.wav --model models/parakeet-tdt-0.6b-v3.q8_0.gguf --device vulkan:0 --json
nemo-speech transcribe clip.wav --model models/nemotron-speech-streaming-en-0.6b.q8_0.gguf --device vulkan:0 --stream
nemo-speech transcribe clip.wav --model models/parakeet-tdt-0.6b-v3.q8_0.gguf --device cpu --json
nemo-speech serve --asr-model models/parakeet-tdt-0.6b-v3.q8_0.gguf --port 8093

docker run --rm -v "$PWD":/audio -v phonon-cache:/home/phonon/.cache \
  ghcr.io/fermionresearch/phonon-cpu:2.0.7 transcribe phonon-2 /audio/clip.wav
uv run --with moonshine-voice moonshine-voice mic --language en

llama-server -hf ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF:Q4_K_M -ngl 99 --port 8095
uv run --with 'openai[realtime]' --with numpy python .runtime/realtime_client.py \
  --url http://127.0.0.1:8095/v1 --type transcription clip16k.wav
```

Check that `words[].confidence` values are meaningful for TDT, and do not assume so from the schema. Compare CPU against Vulkan: small streaming chunks may run faster on CPU. Test clean and phone-line versions of the same clips.

### Post-call ASR

```bash
hf download ggerganov/whisper.cpp ggml-large-v3.bin --local-dir models
docker run --rm --device /dev/dri --group-add $(stat -c %g /dev/dri/renderD128) -v "$PWD/models":/models -v "$PWD":/audio \
  ghcr.io/ggml-org/whisper.cpp:main-vulkan \
  whisper-cli -m /models/ggml-large-v3.bin -f /audio/clip16k.wav -ml 1 -ojf -of /audio/clip

llama-server -hf ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0 -ngl 99 --port 8094
curl http://127.0.0.1:8094/v1/audio/transcriptions \
  -F "model=ibm-granite/granite-speech-5.0-2b-GGUF:Q8_0" -F "file=@clip16k.wav" \
  -F "prompt=transcribe the speech with proper punctuation and capitalization."
```

### TTS

```bash
uvx ptts --lang en --quant q8 --voice Easton "Your invoice is four hundred twelve dollars." -o phonon.wav
uvx ptts --list-voices

uvx --index https://download.pytorch.org/whl/cpu pocket-tts generate \
  --language english_7m_2026-10 --voice alba --text "Your invoice is four hundred twelve dollars."

hf download kyutai/tts-voices --include 'voice-donations/*' 'vctk/*' --local-dir models/tts-voices
# accept terms at https://huggingface.co/kyutai/pocket-tts first, then:
uvx --index https://download.pytorch.org/whl/cpu pocket-tts generate \
  --language english_7m_2026-10 --voice models/tts-voices/voice-donations/<file>.wav --text "Hello."

curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Q4_K_M -p "Hello." --tts-lang en \
  --tts-speaker-file speaker.wav --output qwen.wav -ngl 99
```

For each, measure first-audio latency, RTF with live STT running, and pronunciation of numbers, domains, and `.com`. Phonon's own tokenizer normalizes numbers, so keep text normalization consistent between TTS systems. Check whether Pocket's `--voice` needs a WAV or an exported voice in practice, since cloning from audio is slow and `export-voice` caches it.

### Judge and audio listener

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q6_K.gguf --local-dir models
llama-server -m models/Qwen3.8-27B-UD-Q6_K.gguf --jinja -c 32768 -ngl 99 --port 8096

llama-cli -hf LiquidAI/LFM3-Audio-2B-GGUF:Q8_0 --audio clip.wav \
  -sys "Answer questions about the provided audio clip. Base every answer only on what can be heard." \
  -p "<__media__>Describe the speaker's tone and name any oddly pronounced words." --temp 0

llama-mtmd-cli -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_0 --audio clip_under_30s.wav \
  -p "<__media__>Describe the speaker's tone and name any oddly pronounced words."
```

Plant known errors to test detection: a misread number, a robotic segment, a flat delivery. Score hit rate and false positives before any listener result reaches the dashboard.

### Quality and alignment

```bash
git clone --depth 1 https://github.com/microsoft/DNS-Challenge .runtime/dns
mkdir -p clips/run1 && cp clip16k.wav clips/run1/
(cd .runtime/dns/DNSMOS && uv run --with onnxruntime --with librosa --with soundfile --with pandas --with tqdm --with numpy --with requests \
  python dnsmos_local.py -t "$OLDPWD/clips" -o "$OLDPWD/dnsmos.csv")
uv run --with scoreq python -c "import scoreq; print(scoreq.__file__)"
uv run --with 'git+https://github.com/sarulab-speech/UTMOSv2.git' python -c "import utmosv2; m=utmosv2.create_model(pretrained=True); print(m.predict(input_path='clip16k.wav'))"
```

`dnsmos_local.py` expects one subfolder per system under `-t`. Its ONNX paths should resolve from the `DNSMOS` folder, but I did not read that part of the script. Confirm `uv run --with` does not pull a CUDA PyTorch for UTMOSv2. Keep it in its own environment.
