# Live engine

How mic calls and real-time bot-to-bot calls work. Code is in `src/voicelab/engine/live/`, Pipecat is pinned at 1.15.0, and `engine/service.py` imports it only when a live call starts.

## The pipeline

Both call types run this, and `voice_pipeline` in `parts.py` builds it:

```
transport.input → [mic only: line, recorder, echo gate] → Silero VAD → STT → UserTurnProcessor
  → brain → TTS → [mic only: line] → transport.output → PlayoutTap
```

The order is deliberate:

- **STT sits before the turn processor.** The Smart Turn stop strategy waits for a finalized transcript, so it has to see transcripts.
- **The tap sits after the output.** Pipecat pushes an audio frame past the output transport only once it has been written, so what passes the tap really went out. At a barge-in, queued audio is dropped before it reaches the tap, which makes the tap the measure of what was played.

## Wrappers around our interfaces

One adapter works in every mode because Pipecat only sees thin wrappers.

| Wrapper | Wraps | Notes |
| --- | --- | --- |
| `WrappedSTT` | `STT` | A `SegmentedSTTService`, so `run_stt` gets one VAD segment |
| `WrappedTTS` | `TTS` | `TextAggregationMode.TOKEN`, because the brain already pushes whole sentences |
| `SupportBrain` | `SupportBot` | Replaces the LLM service, so tools, normalization, and `on_interrupted` behave as in turn-based calls |
| `CallerBrain` | `CallerBot` | Hears through its own STT, and decides when to cut in |
| `RealtimeBrain` | `RealtimeVoice` | Replaces both LLM and TTS |

A failure in any of them ends the call as `failed` and keeps what was recorded.

## Barge-in

1. VAD fires and `UserTurnProcessor` broadcasts `InterruptionFrame`. The output drops queued audio and TTS cancels.
2. `BotBrain.on_interrupted` stops the model. For a cascade that closes the `respond` stream, so later sentences are never generated and later tool rounds never run.
3. After `settle_s`, long enough for the chunk being written to finish, the `PlayoutLedger` turns seconds played into words finished. TTS gives no word timing, so a word's share of its sentence follows its characters, and a word still playing at the cut doesn't count.
4. `SupportBot.on_interrupted` rewrites history to those words, and the utterance and `BargeInEvent` go on the timeline.

If the caller's turn then transcribes to nothing, the bot stopped for noise or echo. That's a `false_barge_in` event and a count in `metadata`, the number to watch when tuning.

The browser's jitter buffer holds a little audio past the tap, so the listener heard slightly less than the ledger says. This is not measured.

## Echo

Browser echo cancellation does the first pass. `EchoGate` catches what's left on speakers: it compares the rise and fall of the mic level against the bot's outgoing level across delays up to 400 ms, and blanks mic frames that follow it, so VAD never sees them. It compares level contours rather than waveforms because the codec and speaker change the waveform. A person's speech doesn't follow the bot's contour, so the bot stays interruptible. Sustained echo sends an `echo_suspected` warning to the browser.

Tested on synthetic audio only. It has not been tried on a real speaker in a real room.

## Bot-to-bot

`link.py` joins two pipelines with an in-memory line. Each direction ticks every 20 ms like RTP, sends the next 20 ms of speech or silence through a `PhoneLine`, and delivers it one line delay later. Silence keeps flowing because VAD needs a steady stream to see speech end, and noise has to play between turns. `write_audio_frame` waits for its last tick, so the output paces itself at real time as WebRTC does.

Channels record what came out of the line, the same as turn-based calls, so utterance times are arrival times. The caller's `decide_interruption` starts a timer when it hears the bot begin. When that fires, it transcribes what it has heard so far and speaks over the bot, and the bot's own VAD does the rest, so this is the same barge-in a person triggers.

The caller's pipeline has interruptions off: it only cuts in when told to, and it never stops for the bot.

## Realtime

Profiles with `support_engine: realtime` use `RealtimeBrain`, on the support side only. As a caller, LFM3 slipped into the agent's role.

- The server has no VAD, so Pipecat decides when a turn ends and `commit_turn` asks for the reply.
- It doesn't wait for STT, which runs beside the model and feeds the record and session rebuilds.
- Mic audio only goes to the model during a turn, with 400 ms of pre-roll, because anything sent stays in context.
- Tool calls go through `tool_checks.py` and run one at a time in order. A bad call gets an error as its result.
- On barge-in `Lfm3RealtimeSession.interrupt` closes the socket and opens a new one with heard-only history in `instructions`. Cancelling alone leaves unheard text in context, and the model recites it later.
- Pipecat's `OpenAIRealtimeLLMService` can't be used: it sends `conversation.item.truncate`, the server answers `unknown event type`, and the service stops receiving.

## Spoken ids

Run together, `AC9077` came out of TTS as "AC nineteen seventy seven" and went into STT as "SN977". So:

- `speech.py` spells ids phone-style ("F, as in frank. H, as in house. Four, four, one, nine.") and reads digits as comma-separated words. Measured on ten numbers through real TTS and STT, that came back right 10 of 10 on every voice, against 5 of 10 for spaced numerals on Pocket.
- `heard.py` collapses spelled forms for the brains. The word after "as in" outranks the letter before it, so "S as in Frank" is F. `heard_text` in the artifact stays raw STT output.
- `find_customer` accepts what a recognizer writes. It also compares against how an address sounds, since `nguyenauto.example` is said as "win auto" and no spelling match recovers that. A near match comes back flagged `close_match` so the bot confirms it.

## Hold

A tool turn needs a second LLM round, and Qwen3.5-4B takes up to 2.4 s end to end. If the first round takes longer than `filler_after_s`, `SupportBot` says "One moment, let me check that." before running the tool. It's real speech: in the transcript, in the recording, and in history. In a bot-to-bot call the caller treats that line as a request to wait, because the silence after it otherwise looks like the end of the bot's turn.

## Things to tune

| Setting | Where | Why it's set there |
| --- | --- | --- |
| VAD hang 300 ms | `parts.py` | Measured. Longer adds latency |
| Resume window 0.5 s | `parts.py` | Gaps inside a turn measured 0.32 s median and 0.5 to 0.76 s p90. Smart Turn alone ended turns at sentence gaps, since a finished sentence looks finished. Costs about 0.4 s on every turn |
| `filler_after_s` 0.6 | `LiveSettings` | A tool turn needs a second LLM round, so a slow first round predicts a long silence |
| `settle_s` 0.06 | `LiveSettings` | One output chunk is 40 ms |
| `stop_timeout_s` 5 | `LiveSettings` | Ends a turn that never produced a transcript |
| `caller_patience_s` 12 | `LiveSettings` | A caller left in silence speaks up |
