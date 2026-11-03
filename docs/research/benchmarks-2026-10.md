# Voice stack benchmarks on a Framework Desktop type machine, October 2026

Measured 2026-10-30 and 2026-10-31 on the real machine. Every number below came from a run on this box unless the line says otherwise. Raw results are in `.runtime/bench/results/*.json` (gitignored). `scripts/bench/` reruns everything.

## Verdict

**Can a full voice loop keep up in real time here? Yes, by a wide margin.** The limit in these tests was never compute. In four live two-bot calls the GPU averaged 4 to 23% busy, no audio window was late, and no reply ran dry.

| Stack, all on 8 kHz mu-law input | Plain turn median | Plain max | Tool turn median | Tool max | GPU GB |
| --- | --- | --- | --- | --- | --- |
| LFM3 realtime | 528 ms | 537 ms | 621 ms | 624 ms | 3.2 |
| Parakeet, LFM3-1.7B, Phonon | 681 ms | 737 ms | 1123 ms | 1147 ms | 2.4 |
| Parakeet, Qwen3.5-4B, Phonon | 808 ms | 899 ms | 2395 ms | 2425 ms | 4.8 |
| Qwen3-ASR, Qwen3.5-4B, Phonon | 833 ms | 913 ms | 2001 ms | 2161 ms | 6.6 |

Each number is end of caller speech to first bot audio, and includes 420 to 520 ms of VAD hang.

1. **One human talking to the bot live is viable** on every stack above. Recommended: `local-quality`, Qwen3-ASR plus Qwen3.5-4B plus Phonon, about 830 ms on plain turns. Tool turns need a spoken filler, since they reach 2 s.
2. **Two bots both running live at once is viable. Turn-based is not required.** On the recommended LLM, a 2-minute call ran 11 turns at 1462 ms median and 2192 ms max, with 0 late windows, 0 underruns, GPU 9% mean and 78% peak, CPU 5% mean, 5.5 GB of GPU memory.

Everything loaded at once, the live stack plus LFM3-Audio, Whisper large-v3, and the judge, uses 29.8 GB of GPU memory and leaves 59 GB of system memory free, so no model needs to be swapped out between a call and its evaluation.

The design risks are correctness, not speed:

- Bot-to-bot calls failed on content. No live call reached a refund, because `AC9077` spoken by TTS arrived as "SN977" or "AC nineteen seventy seven". Spell IDs digit by digit in TTS input.
- Tool accuracy from speech varied from 12/12 to 0/12 depending on the STT and LLM pair, because "a C nine zero" reads as an article. Give the account format to the LLM or make `lookup_customer` tolerant.
- A raw 300 ms VAD hang fires inside caller turns, at 6 of 12 turns on these clips. Turn detection needs more than VAD.
- LFM3 realtime cannot honor "context reflects only what was heard" without closing and reopening the session on every barge-in.
- Audio models do not hear delivery. LFM3-Audio, Gemma 4 E2B, and Voxtral all found 0 of 20 planted defects, scoring exactly chance. Use signal checks and ASR diff instead.


## Machine and runtime versions

| Item | Value |
| --- | --- |
| Machine | Framework Desktop type, Ryzen AI MAX+ 395, 16 cores, 32 threads |
| Memory | 128 GB unified, and the iGPU can address almost all of it |
| GPU | Radeon 8060S (`gfx1151`), RADV through Mesa 25.2.8 |
| OS | Ubuntu 25.10, kernel 6.17.0-41, glibc 2.42 |
| llama.cpp | b12017, `0.8.0-dev (build 12017, commit 5047748f1)`, official `llama-b12017-bin-ubuntu-vulkan-x64.tar.gz`, sha256 `e940a901...` |
| NeMo-Speech.cpp | 0.3.0, official `nemo-speech-0.3.0-linux-x86_64-vulkan.tar.gz` |
| whisper.cpp | v1.9.5 (`d1be6fd`), built here with Vulkan. No Linux prebuilt exists |
| ffmpeg | n8.1.3 static, BtbN LGPL build |
| Python packages | numpy, onnxruntime 1.31.1, kokoro-onnx 0.6.1, pocket-tts 4.0.1, ptts 0.6.0, silero-vad, websockets 15.0.1 |
| Pipecat | 1.15.0, source read only |

## Method

**Test set.** `scripts/bench/make_test_audio.py` builds 22 clips, 180 s, under `.runtime/bench/`:

- 12 Kokoro clips with hard content (invoice IDs, amounts, MX, IMAP, domains, spelled account numbers), one female and one male voice. Each has regexes for its critical tokens.
- 8 LibriSpeech clips (CC BY 4.0) and JFK's inaugural clip (public domain).
- One 30 s clip joined from the human speech.
- 5 caller turns for the realtime and end-to-end tests.

Each clip has a telephone copy: ffmpeg to 8 kHz mu-law, then back to 16 kHz. Known limit: all LibriSpeech clips are one male speaker, and the hard clips are synthetic, so ASR accuracy here is a comparison between engines and not an estimate of accuracy on real callers.

**Scoring.** WER after normalizing case, punctuation, and spoken against written numbers. Critical tokens are regexes on the normalized text. A correct `FH-4419` and a correct `F H dash 4 4 1 9` score the same.

**End to end.** `scripts/bench/bench_e2e.py` feeds a clip at wall-clock pace through Silero VAD with a 300 ms hang. The clock starts at the true end of speech, the end of the last 20 ms frame above an energy floor. It stops at the first audio sample ready to send. Network and playout are not included. A 4-turn call runs per repetition so the prompt grows as in a real call. The first repetition warms caches and is left out of the medians.

**Turn detection in these numbers.** VAD ends that fall inside a caller turn are counted and ignored, standing in for a turn-completion model. A raw 300 ms hang fires at sentence gaps. Without that stand-in the bot would start talking over the caller.

**Live two-bot calls.** `scripts/bench/bench_two_bots.py` runs a real call. Audio crosses an 8 kHz mu-law line at wall-clock pace in both directions, each side runs its own VAD on what it hears, and a clock thread checks that 32 ms windows go out on time. "Late windows" counts clock misses. "Underruns" counts a reply that ran dry mid-turn.

**Contamination control.** Benchmarks ran one at a time, and results from the one accidental overlap were discarded and rerun. No other model server had a model loaded, except during the Ollama reference run. Model downloads ran in the background during some runs, using network and disk but under 3% of one core. A browser used about one core at times.

## Results by role

RTF is real-time factor, seconds of audio per second of compute. TTFA is time to first audio. All GPU work is Vulkan unless the row says otherwise.

### LFM3-Audio-2B

It runs fully on Vulkan on this GPU, including the mmproj that holds the audio encoder, depthformer, and codec. The evidence is the speed gap to CPU below and 3.2 GB of GPU memory in use for Q8. No public report existed before this.

| Config | ASR RTF | WER human clean | WER human tel | Critical clean | Critical tel | TTS RTF | TTS first chunk ms | GPU GB | RSS GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-q4 | 16.2 | 0.042 | 0.065 | 28/32 | 28/32 | 4.06 | 91 | 0.0 | 3.0 |
| cpu-q8 | 10.1 | 0.042 | 0.065 | 28/32 | 28/32 | 2.98 | 152 | 0.0 | 3.2 |
| vulkan-q4 | 33.4 | 0.042 | 0.065 | 28/32 | 28/32 | 5.90 | 51 | 2.2 | 0.6 |
| vulkan-q8 | 24.9 | 0.042 | 0.065 | 28/32 | 28/32 | 4.72 | 52 | 3.2 | 0.6 |

`/tts` latency is the first streamed chunk. The first chunk is real audio: 5760 bytes, 60 ms. An audible sample arrives at 76 to 84 ms median. Output is 24 kHz. Voices `female` and `male` measure 197 Hz and 96 Hz in pitch, so they are distinct. An independent ASR heard the `/tts` samples at WER 0 to 0.03 with 24 to 27 of 27 critical tokens.

Realtime through `/v1/realtime`, Q8, Vulkan, 2 runs of 4 clips per mode. Timed from `response.create` to first event:

| `output_modalities` | First text ms | First audio ms | Generation RTF | Notes |
| --- | --- | --- | --- | --- |
| `["text","audio"]` | 40 | 105 | 4.4 | Clean speech. Replies ran long, 6 to 19 s |
| unset, card prompt | 49 | 112 | 4.35 | Same |
| unset, persona in prompt | | | | Silently text only with no speech. Force `["text","audio"]` |
| `["audio"]` | | 46 | 5.1 | Broken. Gibberish, 2 of 8 empty |
| `["text"]` | 40 | | | Emits a `<think>` block first, 4.7 to 16.5 s total. Not usable for live turns |

Long session: 60 turns in one session, 934 s of speech, `-c 32768`, no errors, generation stayed at 4.1x or better. Audio went in faster than real time in that test, so its first-audio numbers are not comparable. Context use per minute was not measured directly, but 15.6 minutes fit in 32K.

Two instances at once, both answering at the same moment, 3 runs each:

| Setup | First audio ms | RTF per session | GPU GB |
| --- | --- | --- | --- |
| One session alone | 104 to 106 | 4.24 to 4.33 | 3.25 |
| One server, `-np 2`, `-c` doubled | 192 to 236 | 2.00 to 2.39 | 3.46 |
| Two server processes | 174 to 178 | 2.64 to 3.06 | 6.50 |

Both stay ahead of real time. Two processes are faster, one server uses half the memory. In a normal call only one side speaks at a time, so this is the worst case.

Listening for tone: see the listener section. It does not hear delivery.

Spoken output needs text normalization for IDs. LFM3 said `AC9077` so that STT heard "AC nineteen seventy seven".

### LLMs for the support bot and caller

Through llama-server chat completions on Vulkan, `--jinja --reasoning off`, a 490 to 720 token system prompt with the three Fernhill tools. Cold is a full prompt evaluation. Warm has the system prompt cached. Scenarios are 4 scripted cases 3 times each: lookup then invoices, refund, a no-tool question, and a refund over the policy limit. Malformed counts arguments that failed JSON parsing. Leaks counts raw call syntax in spoken content.

| Model | Cold TTFT ms | Warm TTFT ms | t/s | Prompt t/s | Scenarios passed | Malformed | Leaks | GPU GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bolt-e4b | 342 | 105 | 56.2 | 1503 | 12/12 | 0 | 0 | 7.4 |
| gemma-e2b | 354 | 54 | 111.3 | 1407 | 6/12 | 0 | 0 | 3.5 |
| lfm3-1.7b | 142 | 131 | 152.6 | 3913 | 12/12 | 0 | 0 | 1.6 |
| qwen3.5-4b | 426 | 92 | 65.5 | 1727 | 12/12 | 0 | 0 | 4.0 |

Tool calls came back correctly formatted on all four models: zero malformed, zero leaks. Gemma 4 E2B did not call tools at all when it should have.

Tool calls from speech, on the same telephone clips, 12 sessions each. This is the test that matters, since it includes STT error. LFM3 realtime hears the audio itself.

| Path | Lookup `AC9077` | get_invoices right id | Refund right | Damaged args |
| --- | --- | --- | --- | --- |
| LFM3 realtime, hears audio directly | 12/12 | 0/12 | 11/12 | 0 |
| parakeet + lfm3-1.7b | 9/12 | 9/12 | 9/12 | 0 |
| parakeet + bolt-e4b | 12/12 | 12/12 | 12/12 | 0 |
| qwen-asr + qwen3.5-4b | 12/12 | 12/12 | 12/12 | 0 |
| parakeet + qwen3.5-4b | 12/12 | 12/12 | 12/12 | 0 |
| whisper + qwen3.5-4b | 0/12 | 0/3 | 0/12 | 0 |

- LFM3 realtime read `AC9077` every time. In forced text+audio mode it fires `get_invoices` in the same response as `lookup_customer`, before the result exists, so it passes the account number as the customer id. It speaks before tools return, so it can state things it has not looked up.
- Elsewhere in LFM3 realtime runs, 2 calls came back damaged: one truncated JSON argument, and one value holding template syntax, `[customer_id]\n</function>...`. Validate every argument.
- With the persona prompt it issued a refund in the lookup turn without being asked.
- LFM3-1.7B took "a C nine zero seven seven" as `C9077` in 3 of 12 sessions.
- These numbers are from 2 clips of one synthetic voice. They rank the stacks. They do not estimate accuracy on real callers.

### Judge LLM

| Model | Runtime | Thinking | t/s | Prompt t/s | Wall s | Valid JSON | Verdicts right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.8-27b-ud-q4 | llama.cpp b12017 Vulkan | off | 13.0 | 271 | 37.5 | 2/2 | 10/10 |
| qwen3.8-27b-ud-q4 | llama.cpp b12017 Vulkan | on | 13.0 | 301 | 98.4 | 2/2 | 10/10 |
| qwen3.8:27b-mtp-q8_0 | Ollama API, read-only | off | 25.4 | 267 | 36.0 | 1/1 | 5/5 |
| qwen3.8:27b-mtp-q8_0 | Ollama API, read-only | on | 22.4 | 565 | 49.4 | 1/1 | 5/5 |

Valid JSON comes from `response_format` with a `json_schema`, so the grammar enforces it, and Ollama's `format` does the same. The test transcript has 5 criteria and 2 planted failures: wrong refund timing and no help with an email problem. All 6 runs got all 5 verdicts right and marked the call as failed. This is one transcript, so it shows the plumbing works, not that the judge is accurate in general.

llama.cpp used 18.7 GB of GPU memory for the UD-Q4_K_M. Time per transcript is 35 to 111 s, which is fine for post-call work. Plan one LLM call per evaluator, not one per criterion.

Ollama's `qwen3.8:27b-mtp-q8_0` ran about 75% faster than the plain Q4 GGUF. It uses a multi-token prediction draft model. llama.cpp can do the same with `ggml-org/Qwen3.8-27B-GGUF` `mtp-Qwen3.8-27B-Q4_0.gguf`, 1.68 GB, which I did not test. Reference run through the API only, and the model was unloaded afterward.

### STT

22 clips, clean and telephone. Median ms is per clip, mean clip length about 8 s. WER is a fraction. Critical is 32 tokens across the 12 hard clips.

| Engine | Median RTF | Median ms | 30 s clip RTF | WER human clean | WER human tel | WER hard clean | WER hard tel | Critical clean | Critical tel | GPU GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parakeet-cpu | 27.5 | 248 | 25.8 | 0.000 | 0.000 | 0.085 | 0.113 | 25/32 | 24/32 | 0.0 |
| parakeet-vulkan | 119.2 | 58 | 187.5 | 0.000 | 0.000 | 0.089 | 0.119 | 24/32 | 25/32 | 0.8 |
| qwen-asr-vulkan | 46.6 | 146 | 53.3 | 0.010 | 0.015 | 0.036 | 0.036 | 29/32 | 29/32 | 2.6 |
| whisper-large-v3-vulkan | 16.9 | 392 | 31.9 | 0.006 | 0.000 | 0.063 | 0.063 | 28/32 | 28/32 | 1.7 |
| whisper-vulkan | 27.1 | 246 | 84.9 | 0.017 | 0.006 | 0.063 | 0.063 | 28/32 | 28/32 | 0.8 |
| lfm3-audio cpu-q4 | 16.2 | 429 | 18.6 | 0.042 | 0.065 | 0.052 | 0.052 | 28/32 | 28/32 | 0.0 |
| lfm3-audio cpu-q8 | 10.1 | 659 | 12.5 | 0.042 | 0.065 | 0.052 | 0.052 | 28/32 | 28/32 | 0.0 |
| lfm3-audio vulkan-q4 | 33.4 | 203 | 40.4 | 0.042 | 0.065 | 0.052 | 0.052 | 28/32 | 28/32 | 2.2 |
| lfm3-audio vulkan-q8 | 24.9 | 272 | 29.2 | 0.042 | 0.065 | 0.052 | 0.052 | 28/32 | 28/32 | 3.2 |

- Parakeet was perfect on human speech and is the fastest by a wide margin, but it loops on repeated digits: "4 4 1 9" became "four four four four four four four one nine", on both Vulkan and CPU. Spoken IDs are the core of this product, so this matters.
- Qwen3-ASR had the best hard-content accuracy. It has no word timestamps through `/v1/audio/transcriptions`, and its text starts with `language English<asr_text>`, which the client must strip.
- Whisper's small-clip time is flat near 250 ms because it always encodes 30 s of mel. Its RTF on long clips is much higher.
- 8 kHz mu-law cost little for any engine on this set. Packet loss and real line noise were not tested here.
- Nothing recovered "IMAP" and "Reyes" reliably without domain hints.
- No engine gives usable word confidence so far. Parakeet's is constant.

### TTS

Same 6 sentences, 2 repetitions, CPU with 16 threads available unless the row says otherwise. Qwen3-TTS on ROCm used 3 sentences with the codec on CPU. It has no streaming, so its TTFA is a full chunk. Critical tokens are 15 across the sentences, checked by Qwen3-ASR hearing the saved audio. It is a proxy: it says the words came out, not that they sound good. I did not listen to any of this audio.

| Engine | RTF | TTFA ms | TTFA max ms | Streams | WER heard | Critical |
| --- | --- | --- | --- | --- | --- | --- |
| kokoro | 7.47 | 768 | 1413 | no | 0.067 | 14/15 |
| lfm3 | 5.99 | 50 | 56 | yes | 0.135 | 12/15 |
| phonon | 10.46 | 23 | 46 | yes | 0.010 | 14/15 |
| pocket-18m | 12.37 | 29 | 31 | yes | 0.087 | 13/15 |
| pocket-2m | 37.76 | 12 | 14 | yes | 0.265 | 4/15 |
| pocket-7m | 21.36 | 20 | 22 | yes | 0.154 | 8/15 |
| pocket-default | 3.81 | 100 | 130 | yes | 0.116 | 12/15 |
| qwen-gguf-vulkan | 2.93 | 1904 | 1962 | yes | 0.010 | 15/15 |
| qwen3-tts-0.6B-rocm | 1.09 | 5688 | 9539 | no | 0.000 | 6/6 |
| qwen3-tts-1.7B-rocm | 0.86 | 6734 | 11621 | no | 0.000 | 6/6 |

Voice counts: Phonon has 102 voices, 37 English. Kokoro has 54, 28 English. LFM3 has 2 on `/tts`, and 1 in realtime. Pocket has 27 presets. Qwen3-TTS CustomVoice has 9.

Pocket preset names hide their sources, read from `pocket_tts/utils/utils.py`:

| Preset | Source | Status |
| --- | --- | --- |
| `marius`, `javert` | `voice-donations/` | CC0, safe |
| `anna`, `vera`, `fantine`, `charles`, `paul`, `eponine`, `azelma`, `george`, `mary`, `jane`, `michael`, `eve` | `vctk/` | CC-BY-4.0, safe with attribution |
| `cosette` | `expresso/` | Non-commercial, do not use |
| `jean` | `ears/` | Non-commercial, do not use |
| `alba`, `bill_boerst`, `peter_yearsley`, `stuart_bell`, `caro_davy`, `estelle` | `alba-mackenna/`, `voice-zero/`, `unmute-prod-website/` | Terms unclear per the research doc, exclude until checked |

Cloning the 228 donated CC0 voices with Pocket needs the gated weights. Qwen3-TTS on Vulkan can clone from a WAV without them, at the speed in the table.

Phonon loads its 64.5 MB model in 71 s on first use here because it downloaded. Pocket first-use load times also include a download. Later loads are short, but they were not timed separately.

### Audio listening

**No model tested beats chance. Do not ship a listener that claims to hear delivery.**

Method, `scripts/bench/bench_listener.py`. Four base clips, two voices, each in six conditions: normal, rushed at 1.7x, dragging at 0.65x, chopped with the audio gated off every other 200 ms, noisy with white noise, robotic with phase removed. That makes 24 trials per question. All were strong, plainly audible edits by design. Trial A is a six-way choice, chance 1/6, so 4 of 24 is the chance score. Trial B asks yes or no whether anything is wrong. Trial C gives the intended text with 4 clips holding a planted wrong word and 4 clean controls: "forty five" for "forty nine", "Fernhole" for "Fernhill", "eye nap" and "mix" for "IMAP" and "MX", "robin" for "refund".

| Model, llama.cpp Vulkan | Six-way correct | p vs chance | Defects found, of 20 | Any-problem hits, of 20 | False alarms, of 4 | Wrong-word hits, of 4 | Wrong-word false alarms, of 4 | Named the word, of 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LFM3-Audio-2B Q8_0 | 4/24 | 0.58 | 0 | 0 | 0 | 4 | 4 | 0 |
| Gemma 4 E2B Q4_0 | 4/24 | 0.58 | 0 | 0 | 0 | 4 | 4 | 2 |
| Voxtral Mini 3B Q4_K_M | 4/24 | 0.58 | 0 | 0 | 0 | 3 | 2 | 1 |
| Signal checks, no model | 22/24 | | 18 | | | | | |
| ASR diff, Whisper turbo | | | | | | 4 | 1 | 4 |
| ASR diff, Parakeet | | | | | | 4 | 2 | 4 |

The models answered "normal" and "no" on nearly every clip, including audio that cut out every 200 ms and audio under loud hiss. The one exception was Gemma picking "rushed" for a chopped clip, which was also wrong. The 4/24 is just the four normal clips. In trial C, LFM3 and Gemma said "no" to all eight clips, so their 4 of 4 hits come with 4 of 4 false alarms and mean nothing. Gemma did name "forty five" correctly, so it hears words, but it also flagged correct clips. Voxtral separated slightly, 3 hits against 2 false alarms, far too few trials to call a signal.

LFM3 was also asked open-ended tone questions. It called flat TTS reading angry words "frustrated" and "upset", and rated rushed, dragging, chopped, noisy, and robotic clips "calm" and "neutral". It reads the words, not the voice.

Limits of this result. Only three models ran. MOSS-Audio-4B, the best candidate on public numbers, could not be tested here, and Qwen3-Omni did not fit the budget. Only 2 question wordings ran, on synthetic base voices. A different wording might do better, but a model that misses gated audio under a direct question should not be trusted on subtler problems.

The signal row shows what to use instead. Clip duration against the same text gives pace, the noise floor gives noise, and dropout count gives chopping. It found 18 of 20 defects with no false alarms on the normals, missing 2 robotic clips. I set those thresholds with this data in view and they use a clean baseline of the same sentence, so 22/24 is an illustration and not a validated detector. The wrong-word row shows what the pronunciation evaluator in the brief can do with an independent ASR.

Recommendation: ship no audio-model listener for delivery. Use signal checks for pace, noise, and dropouts, a MOS predictor for quality, and ASR diff for wrong words. If the dashboard keeps a listener, label it as experimental, off by default, and unable to detect these faults.

### Memory

| Loaded so far | GPU total GB | Process RSS GB | System used, change GB |
| --- | --- | --- | --- |
| qwen3-asr 0.6b q8 -np 2 | 2.6 | 0.16 | 3.2 |
| qwen3.5-4b q4_k_m -np 2 | 7.1 | 0.20 | 7.9 |
| lfm3-audio q4_k_m | 9.4 | 0.18 | 10.3 |
| phonon in this process | 9.4 | n/a | 10.7 |
| pocket 18m in this process | 9.4 | n/a | 10.4 |
| whisper large-v3 q5 | 11.1 | 0.10 | 12.1 |
| judge llm | 29.7 | 0.22 | 25.5 |
| after one request each | 29.8 | n/a | 26.0 |

Free system memory with everything loaded: 59 GB.

GPU numbers are the system-wide GTT plus VRAM counters, measured with everything from `local-quality`, the LFM3 alternative, eval STT, and the judge resident. System figures move with other work on the box, so read them as approximate. Phonon and Pocket ran in the measuring process, so their RSS is included in the system column only. `local-fast` alone needs 2.4 GB of GPU memory.

## End to end

End of caller speech to first bot audio, telephone audio, Silero VAD with a 300 ms hang, 8 of 12 turns after the warm-up pass. Tool turns are `caller-lookup` and `caller-refund`.

| Stack | Plain median ms | Plain max | Tool median ms | Tool max | GPU GB |
| --- | --- | --- | --- | --- | --- |
| lfm3-realtime | 528 | 537 | 621 | 624 | 3.2 |
| parakeet, lfm3-1.7b, phonon | 681 | 737 | 1123 | 1147 | 2.4 |
| parakeet, lfm3-1.7b, lfm3 | 757 | 820 | 1189 | 1276 | 4.6 |
| parakeet, qwen3.5-4b, pocket | 797 | 898 | 2384 | 2575 | 4.8 |
| parakeet, qwen3.5-4b, phonon | 808 | 899 | 2395 | 2425 | 4.8 |
| qwen-asr, qwen3.5-4b, phonon | 833 | 913 | 2001 | 2161 | 6.6 |
| parakeet, qwen3.5-4b, lfm3 | 841 | 1010 | 2391 | 2571 | 6.9 |
| parakeet, lfm3-1.7b, kokoro | 1180 | 1459 | 1653 | 2264 | 2.4 |

VAD is the biggest single cost in every row: 420 to 520 ms for a 300 ms hang, or 473 ms mean. A shorter hang or a turn-completion model moves every row by the same amount. Smart Turn was not measured.

Per stage, plain turns: STT 27 to 60 ms for Parakeet, about 100 ms for Qwen3-ASR. LLM to first sentence 136 to 240 ms for LFM3-1.7B, 230 to 420 ms for Qwen3.5-4B. TTS first audio 20 to 35 ms for Phonon and Pocket, 75 to 100 ms for LFM3 `/tts`, and 270 to 990 ms for Kokoro.

Tool turns are the weak spot of Qwen3.5-4B. Reaching the first spoken sentence takes the tool call and then a new LLM pass, 1.8 s of LLM time on a successful turn at 65 t/s. That gives about 2.4 s end to end with Parakeet. The Qwen3-ASR row reads lower only because its lookup failed and the turn ended after one short round. The engine should play a short filler if no sentence has arrived by about 700 ms. LFM3-1.7B stays near 1.1 s because it generates at 153 t/s.

## Live two-bot calls

Audio crossing the 8 kHz mu-law line at wall-clock pace in both directions. Every call ran for 2 minutes.

| Stack, support then caller TTS | Turns | E2E median ms | E2E max ms | Late windows | Underruns | GPU busy mean/peak % | CPU mean/peak % | GPU GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parakeet, Qwen3.5-4B, Phonon, Pocket 7M | 11 | 1462 | 2192 | 0 | 0 | 9 / 78 | 5 / 66 | 5.5 |
| Parakeet, LFM3-1.7B, Phonon, Pocket 7M | 8 | 1117 | 1311 | 0 | 0 | 4 / 58 | 5 / 66 | 2.8 |
| Parakeet, LFM3-1.7B, LFM3 `/tts` both | 5 | 1157 | 1410 | 0 | 0 | 19 / 87 | 5 / 39 | 5.3 |
| LFM3 realtime, two processes | 10 | 557 | 648 | 0 | 0 | 23 / 85 | 6 / 26 | 6.7 |

These calls used Pocket 7M for the caller because they ran before the accuracy scoring. Use Pocket 18M instead: it has 13/15 critical tokens against 8/15 and still runs 12.4x, but it was not run in a live call.

The E2E clock here starts at the last window with speech energy on the line, which runs higher than the single-side numbers since TTS tails and the line itself delay the VAD. These are 1 call each with 5 to 11 turns, so treat the medians as rough.

Hardware was never the limit. The calls did show content failures that matter more:

- **IDs do not survive TTS plus STT.** The caller's TTS said `AC9077`, and support's STT heard "SN977", "SE ninety seven seven", "AC nineteen seventy seven". No call on any stack reached a refund for this reason. The caller bot must spell IDs digit by digit before TTS.
- **LFM3 realtime as a caller slips into the agent role.** It answered "I understand you are following the required process". Do not use it for caller personas.
- **Callers drift with a fact sheet alone.** LFM3-1.7B and Qwen3.5-4B callers apologized, invented lines, and at times spoke as the agent. The brief's ordered beats are needed.


## `/v1/realtime` protocol notes

Observed against llama.cpp b12017 with `scripts/bench/rt_client.py`, a raw `websockets` client. Raw event logs are in `.runtime/bench/results/rt-protocol-vulkan-q8.json`. Source: `tools/server/server-realtime.cpp` at b12017.

### Connect

`ws://host:port/v1/realtime`. The first event is `session.created` once the session holds a slot. If every slot is busy the connection opens and then waits with no event and no error. A queued connection keeps its place in line, so a client that gives up must close it, or it takes the slot when one frees.

```json
{"id":"sess_...","object":"realtime.session","type":"realtime","model":"...","instructions":"",
 "output_modalities":null,"max_output_tokens":"inf","tools":[],"tool_choice":"auto",
 "audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"turn_detection":null},
          "output":{"format":{"type":"audio/pcm","rate":24000}}}}
```

Input is base64 PCM16 mono at 16 kHz. Output is base64 PCM16 mono at 24 kHz.

### Client events

| Event | Observed behavior |
| --- | --- |
| `session.update` | Sets `instructions`, `tools`, `tool_choice`, `output_modalities`, `max_output_tokens`. Replies `session.updated`. `instructions` and `tools` lock after the first audio or message: the server sends `error`, then still sends `session.updated`. `output_modalities` can change later |
| `input_audio_buffer.append` | Base64 PCM. No reply. The audio goes straight into the conversation as it arrives |
| `input_audio_buffer.commit` | Ends the user turn. Replies `input_audio_buffer.committed`, `conversation.item.added`, `conversation.item.done` |
| `input_audio_buffer.clear` | Replies `input_audio_buffer.cleared` and removes nothing. The next response still answered the audio sent before the clear |
| `conversation.item.create` | `role: user` with `input_text`, or `function_call_output`. Anything else returns `error`: `only user messages and function call outputs are supported`. That covers `assistant` and `system` roles |
| `response.create` | Commits any uncommitted audio and generates. Accepts `response.output_modalities`, `max_output_tokens`, `tool_choice` |
| `response.cancel` | Stops generation. Silent no-op with no event if nothing is running |
| `conversation.item.truncate`, `.delete`, `.retrieve`, `output_audio_buffer.clear` | `error`: `unknown event type: ...`, `code: null` |

### Server events for one response

With `output_modalities: ["text","audio"]`, in order:

1. `input_audio_buffer.committed`, `conversation.item.added`, `conversation.item.done`. The user item holds only `{"type":"input_audio"}`, with no transcript
2. `response.created`
3. `response.output_item.added`, role assistant
4. `response.content_part.added` for `output_text`
5. `response.output_text.delta`, the first ones arrive about 40 ms after `response.create`
6. `response.content_part.added` for `output_audio`
7. Interleaved `response.output_text.delta` and `response.output_audio.delta`, the first audio about 105 ms after `response.create`
8. `response.output_text.done`, `response.content_part.done`, `response.output_audio.done`, `response.content_part.done`, `response.output_item.done`, `response.done` with `status` and `usage.output_tokens`

The `output_audio` part has an empty `transcript`. The spoken text is in the `output_text` part.

Tool calls add `response.output_item.added` with a `function_call` item, `response.function_call_arguments.delta`, and `response.function_call_arguments.done`. The client sends `function_call_output` and then a new `response.create`.

### What the API does not have

| Capability | Result |
| --- | --- |
| Server VAD | None. Speech plus 2 s of silence with no commit produced zero events. `turn_detection` is always `null`. No `speech_started` or `speech_stopped` |
| Truncate | None. See below |
| Assistant message injection | Refused |
| Voice selection | None. `audio.output.voice`, a top-level `voice`, and "using a male voice" in the prompt all left pitch at 183 to 242 Hz. `/tts` has both voices, realtime has one |
| User transcript | None in a `realtime` session. Sessions opened as `transcription` do transcribe, each committed buffer on its own |
| Removing audio | None. Audio is in the context on arrival |

### Interruption

`response.cancel` works and is fast. Over 4 runs, `response.done` with `status: cancelled` came 17 to 30 ms after the cancel, with 2 more audio deltas in that gap.

The problem is what stays in context. The server writes speech far faster than real time, so at the moment of a barge-in the model has generated much more than the caller heard. After 1 s of playback, the session held 4.2 s of generated audio and 144 to 162 characters of text. With no truncate, all of it stays in context. A follow-up question got the whole generated text back, including words nobody heard.

Tests of ways around this:

| Method | Result |
| --- | --- |
| User text note: "I only heard your reply up to ..." | Failed, 2 of 2. The model recited the unheard text anyway |
| Close the session, open a new one with heard-only history in `instructions` | Worked, 4 of 4. The model said only the heard words. New session ready 4 to 6 ms after close, first text 64 to 70 ms after `response.create` |

The session rebuild has costs. The client must keep the transcript itself, and the realtime session supplies no user transcript, so it needs a separate STT stream on the caller audio. Earlier caller audio leaves the context, so the model loses anything the transcript got wrong. `tools` must be sent again. The rebuild test used a short text history. A long call's history in `instructions` was not tested.

Audio appended during a running response waits in the slot's queue. It is not processed until the response ends or is cancelled, and it then counts as the next user turn.


### Pipecat

Pipecat's `OpenAIRealtimeLLMService` in 1.15.0 cannot drive this server unchanged. `_handle_interruption()` calls `_truncate_current_audio_response()`, which sends `conversation.item.truncate`. The server answers with `error` and `code: null`. The receive loop only ignores `response_cancel_not_active` and `conversation_already_has_active_response`. Anything else reaches `_handle_evt_error`, followed by the comment "errors are fatal, so exit the receive loop". The service stops receiving at the first interruption. It needs its own small service class.

## What failed and why

| What | Result | Cause or action |
| --- | --- | --- |
| `https://llama.app/install.sh` | Not run | It deletes and writes `~/.llama-app` and installs to `~/.local/bin`. The official GitHub tarball does the same job inside `.runtime/` |
| whisper.cpp prebuilt | None for Linux | Built v1.9.5 with a conda-forge toolchain in `.runtime/toolchain` from micromamba: gcc 14, cmake, ninja, shaderc, glslang, Vulkan headers and loader, spirv-headers. No system packages |
| MOSS-Audio-4B | Not tested | The only non-CUDA path found is CrispASR, a third-party binary. The permission system refused to run it, so I did not pursue it. llama.cpp has no projector for it. Partial downloads were removed. It stays the main untested listener candidate |
| Qwen3-Omni 30B | Not tested | 19.9 GB, over the download budget |
| Pocket TTS voice cloning | Blocked | `kyutai/pocket-tts` is gated and this HF account gets 403. The user has to accept the terms. I did not use third-party re-uploads of those weights. Preset voices work without it |
| `output_modalities: ["audio"]` | Broken | Audio comes back as gibberish ("Sarosiomanzi"), sometimes empty or 30 s long. Reproduces llama.cpp issue #31436. Use `["text","audio"]` |
| Gemma 4 E2B as support LLM | Rejected | 0 of 6 scenarios that need a tool. It answered in text and once claimed a refund it never issued |
| Pocket 2M and 7M for the bot | Rejected | Fast, but 4/15 and 8/15 critical tokens. They mangle "Fernhill" |
| Kokoro for live use through `kokoro-onnx` `create()` | Works, slow to start | No sub-sentence streaming through this call, so first audio is a whole sentence: 768 ms median, 1413 ms max |
| Qwen3.5-9B, Gemma 4 E4B, Qwen3.6-35B-A3B | Not downloaded | Out of budget once Bolt and Qwen3.8 were requested. `scripts/bench/download_models.sh extra` fetches them |
| Parakeet `words[].confidence` | Useless | Exactly 1.0 on all 265 words checked, including wrong ones. Timestamps are present |

Harness mistakes corrected along the way, listed so nobody reuses the early numbers:

- The first two-bot cascade run waited for whole TTS sentences, giving a false 6 s worst case. Streaming fixed it.
- The second started its clock at the last sample on the line, including TTS tail silence, which read low. Speech end is now energy-based. Only the corrected runs appear below.
- VAD ends that landed in TTS tail silence stalled one run at 0 turns. A real engine has the same race: treat a VAD end as pending until the far side has finished sending.
- `pkill -f` matched its own shell twice. Use `pgrep` with a bracketed pattern.

## Recommended stacks

Licenses below were read from each repository's metadata or license file on 2026-10-31.

### `local-quality`

The support bot and caller for normal use. Live is viable.

| Role | Choice | Numbers | Code license | Weights license |
| --- | --- | --- | --- | --- |
| Live STT | Qwen3-ASR 0.6B Q8_0, llama.cpp Vulkan | 146 ms, 46x, 29/32 critical, 2.6 GB | MIT | Apache-2.0 |
| LLM, support and caller | Qwen3.5-4B Q4_K_M, llama.cpp Vulkan, `-np 2` | 92 ms warm TTFT, 65.5 t/s, 12/12 tool sessions from speech, 4.0 GB | MIT | Apache-2.0 |
| Support TTS | Phonon 31m q8, `ptts` on CPU | 23 ms TTFA, 10.5x, 14/15 critical, streams | MIT or Apache-2.0 | Apache-2.0 |
| Caller TTS | Phonon catalog first, Pocket 18M presets from the safe list for more voices | Pocket: 29 ms, 12.4x, 13/15 | MIT | Apache-2.0, CC-BY-4.0, CC0 voices |
| VAD | Silero, 300 ms hang measured | 0.6 ms per 32 ms window, CPU | MIT | MIT |
| Eval STT | Whisper large-v3 q5_0, whisper.cpp Vulkan | 16.9x on short clips, 22x on a 300 s clip, WER 0.006 on human speech, 28/32 critical | MIT | MIT |
| Judge | Qwen3.8-27B UD-Q4_K_M, llama.cpp Vulkan | 13 t/s, 35 to 111 s per call, 4/4 valid JSON, 20/20 verdicts, 18.7 GB | MIT | Apache-2.0 |

End to end: plain turn 833 ms median, 913 ms max. Tool turn 2.0 s median, so add the filler line.

Parakeet can replace Qwen3-ASR if GPU memory or word timestamps matter more than spoken IDs. It scored the same 12/12 here, saves 1.8 GB, and is 90 ms faster, but it has the digit-loop defect.

### `local-fast`

The lowest latency cascade and the smallest GPU footprint, at the cost of tool reliability.

| Role | Choice | Numbers | Code license | Weights license |
| --- | --- | --- | --- | --- |
| Live STT | Parakeet TDT 0.6B v3 Q8_0, NeMo-Speech.cpp 0.3.0 Vulkan | 58 ms, 120x, 0.8 GB | Apache-2.0 | CC-BY-4.0 |
| LLM | LFM3-1.7B Q4_K_M, llama.cpp Vulkan, `-np 2` | 131 ms warm TTFT, 153 t/s, 9/12 tool sessions from speech, 1.6 GB | MIT | LFM Open License v1.0, see below |
| TTS | Phonon 31m q8 for both sides | as above | MIT or Apache-2.0 | Apache-2.0 |

End to end: plain turn 681 ms median, 737 ms max. Tool turn 1123 ms median, 1147 ms max. GPU 2.4 GB.

### Slower TTS for turn-based calls and persona voices

Qwen3-TTS 1.7B through `llama-server` `/tts` on Vulkan had the best accuracy of any engine, 15/15, and clones from a WAV with no gated weights. At 2.9x real time it keeps up, but first audio is 1.9 s, so it does not fit a live turn. It suits turn-based runs and pre-generated persona reference clips. It is over 3x faster than the same model on ROCm, so use Vulkan and not the `qwen-tts` package. Apache-2.0 weights, MIT code. Run it with `-m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf -mm mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf -c 4096 -np 1 -ngl 999` and pass `speaker_ref_b64` to `/tts`.

### Alternative: LFM3 realtime

`LFM3-Audio-2B` through `/v1/realtime`, two server processes. It is the fastest end to end, 528 ms plain and 621 ms on tool turns, and it is the only path that reads IDs from audio without an STT step. It needs its own Pipecat service, your own VAD, forced `["text","audio"]`, argument validation, and the session rebuild for interruptions. It has one voice and no user transcript. It should not play the caller.

**LFM license terms.** LFM Open License v1.0 is Apache 2.0 with a Section 5: commercial use is licensed only while the using entity has under USD 10 million in annual revenue, with a carve-out for qualified non-profits doing non-commercial or research work. This is fine for this project. A larger company adopting the design would need a separate license from Liquid AI, which is why `local-quality` avoids LFM models entirely.

### Exact launch commands

`scripts/bench/launch_stack.sh local-quality`, `local-fast`, `judge`, and `stop` run these. Ports are in the 18xxx range. Paths are relative to `.runtime/`.

```bash
export LD_LIBRARY_PATH=$PWD/llama-vulkan/llama-b12017

# local-quality STT, port 18101
llama-vulkan/llama-b12017/llama-server --host 127.0.0.1 --port 18101 --no-webui \
  -m models/ggml-org__Qwen3-ASR-0.6B-GGUF/Qwen3-ASR-0.6B-Q8_0.gguf \
  -mm models/ggml-org__Qwen3-ASR-0.6B-GGUF/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --temp 0 -c 8192 -np 2 -ngl 999

# local-quality LLM, port 18102
llama-vulkan/llama-b12017/llama-server --host 127.0.0.1 --port 18102 --no-webui \
  -m models/unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf \
  --jinja --reasoning off -c 32768 -np 2 -ngl 999

# local-fast STT, port 18101. The model field in requests must be parakeet-tdt-0.6b-v3.oss-align.q8_0
LD_LIBRARY_PATH=$PWD/nemo/nemo-speech-0.3.0-linux-x86_64-vulkan/lib \
nemo/nemo-speech-0.3.0-linux-x86_64-vulkan/bin/nemo-speech serve --no-ui --port 18101 \
  --asr-model models/nvidia__parakeet/parakeet-tdt-0.6b-v3.q8_0.gguf --device vulkan:0

# local-fast LLM, port 18102
llama-vulkan/llama-b12017/llama-server --host 127.0.0.1 --port 18102 --no-webui \
  -m models/LiquidAI__LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf \
  --jinja --reasoning off -c 32768 -np 2 -ngl 999 --temp 0.1 --top-k 50 --repeat-penalty 1.1

# LFM3-Audio for /tts, or one process per realtime side, port 18103
llama-vulkan/llama-b12017/llama-server --host 127.0.0.1 --port 18103 --no-webui \
  -m models/LiquidAI__LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q4_K_M.gguf \
  -mm models/LiquidAI__LFM3-Audio-2B-GGUF/mmproj-LFM3-Audio-2B-Q4_K_M.gguf \
  --temp 0 -c 16384 -np 2 -ngl 999

# eval STT, port 18104. HTTP route is POST /inference
LD_LIBRARY_PATH=$PWD/toolchain/lib whisper.cpp/build/bin/whisper-server \
  -m models/whisper/ggml-large-v3-q5_0.bin --host 127.0.0.1 --port 18104 -l en -t 8

# judge, port 18105
llama-vulkan/llama-b12017/llama-server --host 127.0.0.1 --port 18105 --no-webui \
  -m models/unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --jinja --reasoning auto -c 32768 -np 1 -ngl 999 --temp 0.1
```

In-process TTS:

```python
import ptts
tts = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant="q8")
with tts.stream(text, voice="Easton") as audio:
    for pcm in audio:   # float32 chunks at audio.sample_rate, 24 kHz
        ...

from pocket_tts import TTSModel
model = TTSModel.load_model(language="english_18m_2026-10")
voice = model.get_state_for_audio_prompt("marius")
for chunk in model.generate_audio_stream(voice, text):
    ...                 # not thread-safe, one instance per side
```

Rerun everything:

```bash
.runtime/venvs/bench/bin/python scripts/bench/make_test_audio.py
cd scripts/bench
PY=../../.runtime/venvs/bench/bin/python
$PY bench_stt.py parakeet-vulkan parakeet-cpu whisper-vulkan whisper-large-v3 qwen-asr-vulkan
$PY bench_tts.py kokoro phonon pocket-18m pocket-7m pocket-2m pocket-default lfm3 && $PY bench_tts.py --score
$PY bench_lfm_asr_tts.py vulkan-q8 vulkan-q4 cpu-q8 cpu-q4
$PY bench_lfm_realtime.py protocol latency tools cancel concurrent persona heard workaround probes long toolacc listen
$PY bench_llm.py lfm3-1.7b qwen3.5-4b gemma-e2b bolt-e4b
$PY bench_toolacc.py cascade qwen3.5-4b --stt parakeet
$PY bench_e2e.py parakeet+qwen3.5-4b+phonon lfm3-realtime --reps 3
$PY bench_two_bots.py cascade --stt parakeet --llm qwen3.5-4b --tts phonon --tts-caller pocket
$PY bench_listener.py make && $PY bench_listener.py signal && $PY bench_listener.py report
$PY summarize.py
```

Setup steps are in `scripts/bench/download_models.sh` and `scripts/bench/build_whisper_cpp.sh`. The bench venv needs: `numpy soundfile websockets httpx "openai[realtime]" jiwer psutil onnxruntime kokoro-onnx onnx-asr sherpa-onnx pocket-tts silero-vad pyarrow ptts scipy`.

## Answers to open questions in the research doc

- LFM3-Audio on Vulkan on an AMD GPU: works, on RADV with Mesa 25.2.8.
- Two realtime sessions on one server: works at 2.0x to 2.4x. Two processes do better at 2.6x to 3.1x.
- `response.cancel` flush: 2 more audio deltas arrive after the cancel, within 17 to 30 ms.
- Audio-only mode: issue #31436 reproduced here.
- `words[].confidence` in NeMo-Speech.cpp: constant 1.0, no information.
- NeMo-Speech.cpp v0.3.0 Vulkan tarball on RADV: works as shipped, 0.8 GB for Parakeet.
- x86 speed: Phonon 31m 10.5x, Pocket 18M 12.4x, Pocket 7M 21.4x, Pocket 2M 37.8x, Kokoro 7.5x, all with 16 threads available.
- Gemma Vulkan prefill anomaly: not seen. E2B prefilled at 1407 t/s.
- Download total for this work: 38 GiB under `.runtime/models`, plus 0.5 GiB of Phonon and Pocket weights in `.runtime/hf`. That is over the 35 GB target and near the 40 GB limit, because Whisper large-v3 and the listener models were added late.

## Not verified

- Real human callers, accents, and real phone lines. One LibriSpeech speaker and one JFK clip are the only human speech.
- Packet loss, added noise, and the full simulated line beyond mu-law.
- Echo and speakers. Barge-in timing through Pipecat. Every interruption test here was at the protocol level.
- Smart Turn or any turn-completion model. The stand-in here assumes it never fires early.
- Voice quality. I checked intelligibility with ASR. Nobody listened.
- More than one two-sided call at once, or eval running during a live call.
- Nemotron streaming ASR, downloaded but not run. Partial transcripts might shorten STT further.
- llama.cpp ROCm build, which the research doc flags as possibly lower latency.
- Live ASR on LFM3-Audio, `?intent=transcription`.
