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
4. `SupportBot.on_interrupted` rewrites history to those words, and the utterance and `BargeInEvent` go on the timeline. Only finished sentences stay in the bot's own turn. The words of the sentence it was cut off in go in front of the caller's next turn, as a note. Qwen3.5-4B copies its last turn: history ending "two charges of forty nine" made it end later replies at "forty nine", 7 times in 8, and every marker put on that text got copied too. With the note, none were cut in 16 trials.

Hanging up during a reply, or cancelling a live call, follows the same steps: stop the model, settle, and record what went out as an interrupted utterance, then finalize the artifact. Recording never waits, so a second cancel during settling can't lose the turn.

`UserStoppedSpeakingFrame` is a system frame and a transcript isn't, so when STT is what the turn was waiting for, the end of the turn overtakes its words. The brain waits up to 0.25 s for them. Without that, a 6 s sentence with the GPU busy looked like an empty turn, and the bot never answered it.

If the caller's turn then transcribes to nothing, the bot stopped for noise or echo. That's a `false_barge_in` event and a count in `metadata`, the number to watch when tuning.

What counts as heard is an estimate made at the server. The tap measures transmission, not playback: a frame passes it when aiortc takes its last 10 ms for encoding, and nothing reports back what the listener's device played. After the tap come Opus framing, the network, the client's jitter buffer, and its output device.

The ledger allows for that with `in_transit_s`, taken off played time before words are credited. Live bot calls use the line's latency, which is known exactly. Mic calls use `LiveSettings.client_buffer_s`, 80 ms, from one run of Chrome 154 headless over loopback with the fake profile: 34 ms in the jitter buffer (`jitterBufferDelay` over `jitterBufferEmittedCount`, steady across 11 seconds), 24 to 34 ms of `AudioContext.outputLatency` on a null audio device, and about 10 ms of Opus framing on average. That's a fixed figure. It wasn't measured on a real network, a real sound card, Bluetooth, Firefox, or Safari, and any of those could hold more, in which case the ledger credits a word too many. When a cut lands in a gap between sentences, the whole allowance still comes off even though the audio has already arrived, so the last word of the sentence before can go uncredited.

## Echo

Browser echo cancellation does the first pass. `EchoGate` catches what's left on speakers. It compares eight band levels of the mic against the bot's outgoing audio, every 10 ms over a 400 ms window, across delays up to 400 ms, and blanks mic frames that follow the bot, so VAD never sees them. It compares band contours rather than waveforms because the codec and speaker change the waveform. Sustained echo sends an `echo_suspected` warning to the browser and counts blanked frames in `metadata`.

The output sends nothing between replies, so the mic is the detector's clock. Time the output stayed quiet goes into the outgoing contour as silence, and echo is only watched for until 300 ms after the bot's last audio. VAD fires at 200 ms, sooner than a contour of speech can fill, and silence followed by a rise is already a shape to compare. That lets the same check work from the first steps of a reply. Nothing compares how loud the two sides are, because a quiet person isn't echo.

Measured by replaying real Phonon and Pocket audio from `local-default` calls, with room reverb on the echo path (direct sound plus a 120 ms decaying tail) and delays of 60 to 300 ms:

| Case | Result |
|---|---|
| Echo at gain 0.05, 0.2, 0.4 | No run of 200 ms unblanked in 45 cases. 94% of frames blanked |
| Person, no echo underneath | 1% of their frames blanked |
| Person over echo at gain 0.1, person as loud or louder | 3 to 15% blanked, 200 ms clear within 70 ms at the median |
| Person at half the echo's level | 18 to 35% blanked, 200 ms clear within 180 ms at the median |

A person starting within 400 ms of a bot rise can match it for a moment, and someone quieter than the echo loses some of their speech. They aren't lost for the whole turn, and the recording keeps them either way, since `MicRecorder` runs before the gate.

Replay and synthetic audio only. It has not been tried on a real speaker in a real room.

## Bot-to-bot

`link.py` joins two pipelines with an in-memory line. Each direction ticks every 20 ms like RTP, sends the next 20 ms of speech or silence through a `PhoneLine`, and delivers it one line delay later. Silence keeps flowing because VAD needs a steady stream to see speech end, and noise has to play between turns. `write_audio_frame` waits for its last tick, so the output paces itself at real time as WebRTC does.

Channels record what came out of the line, the same as turn-based calls, so utterance times are arrival times. The caller's `decide_interruption` starts a timer when it hears the bot begin. When that fires, it transcribes what it has heard so far and speaks over the bot, and the bot's own VAD does the rest, so this is the same barge-in a person triggers.

The caller's pipeline has interruptions off: it only cuts in when told to, and it never stops for the bot.

## Realtime

Profiles with `support_engine: realtime` use `RealtimeBrain`, on the support side only. As a caller, LFM3 slipped into the agent's role.

- The server has no VAD, so Pipecat decides when a turn ends and `commit_turn` asks for the reply. It asks only if audio reached the current socket since the last ask. The server treats anything else as an error, and that comes up when a pause ended a turn and a cut-in then replaced the session.
- The pipeline reports started before the session is up, so `greet` waits for the connection to settle. Skipping that left both sides silent until the call timed out.
- The server holds one session per slot and this profile has one, so `max_calls: 1` caps a run at one call at a time.
- It doesn't wait for STT, which runs beside the model and feeds the record and session rebuilds.
- Mic audio only goes to the model during a turn, with 400 ms of pre-roll, because anything sent stays in context.
- Tool calls go through `tool_checks.py` and run one at a time in order. A bad call gets an error as its result.
- On barge-in `Lfm3RealtimeSession.interrupt` closes the socket and opens a new one with heard-only history in `instructions`. Cancelling alone leaves unheard text in context, and the model recites it later.
- The model finishes well before its audio plays, so a reply goes in the history when generated and a cut trims it. Played time counts from the start of the turn, across every reply in it. Text that outran its audio is timed at a speaking rate until the reply is done.
- Events queued when a cut lands are dropped, by session epoch and while the cut settles, so a cancelled response's tools never run.
- Pipecat's `OpenAIRealtimeLLMService` can't be used: it sends `conversation.item.truncate`, the server answers `unknown event type`, and the service stops receiving.

## Spoken ids

Run together, `AC9077` came out of TTS as "AC nineteen seventy seven" and went into STT as "SN977". So:

- `speech.py` spells ids phone-style ("F, as in frank. H, as in house. Four, four, one, nine.") and reads digits as comma-separated words. Measured on ten numbers through real TTS and STT, that came back right 10 of 10 on every voice, against 5 of 10 for spaced numerals on Pocket.
- `heard.py` collapses spelled forms for the brains. The word after "as in" outranks the letter before it, so "S as in Frank" is F. `heard_text` in the artifact stays raw STT output. The eval tokenizer collapses a spelled letter only when it agrees with its word. A contradiction like "F as in Sam" matches neither F nor S, so it shows up as a mismatch on the id.
- Lookups accept what a recognizer writes, such as "bell hardware dot example". They also compare against how an address sounds, since `nguyenauto.example` is said as "win auto" and no spelling match recovers that. `find_customer` is the one tolerant lookup: on a near match it returns the account with `did_you_mean`, the exact address. The bot confirms the name on the account and goes through zip and PIN as usual. Confirming by name matters. In a real call the bot read back "marino cafe" correctly, the caller's STT heard "merino", and it refused its own domain three times, because the two sound alike. Every other tool takes only the exact address and refuses a near one with the same suggestion, so no write lands on a guess, and matches that span two customers aren't offered at all.

## Hold

A tool turn needs a second LLM round, and Qwen3.5-4B takes up to 2.4 s end to end. If the first round takes longer than `filler_after_s`, `SupportBot` says "One moment, let me check that." before running the tool. It's real speech: in the transcript, in the recording, and in history. In a bot-to-bot call the caller treats that line as a request to wait, because the silence after it otherwise looks like the end of the bot's turn.

## Things to tune

| Setting | Where | Why it's set there |
| --- | --- | --- |
| VAD hang 300 ms | `parts.py` | Measured. Longer adds latency |
| Resume window 0.5 s | `parts.py` | Gaps inside a turn measured 0.32 s median and 0.5 to 0.76 s p90. Smart Turn alone ended turns at sentence gaps, since a finished sentence looks finished. Costs about 0.4 s on every turn |
| Smart Turn unsure wait 1 s | `parts.py` | Smart Turn called a finished sentence unfinished on 2 of 7 turns of a real mic call. Pipecat's default of 3 s turned each into a 5 to 6 s reply. Barge-in covers the opposite mistake |
| `filler_after_s` 0.6 | `LiveSettings` | A tool turn needs a second LLM round, so a slow first round predicts a long silence |
| `settle_s` 0.06 | `LiveSettings` | One output chunk is 40 ms |
| `stop_timeout_s` 5 | `LiveSettings` | Ends a turn that never produced a transcript |
| `caller_patience_s` 12 | `LiveSettings` | A caller left in silence speaks up. Pipecat's idle timer only runs from that side's own speech, so `CallerBrain` keeps its own for when it owes a reply |
