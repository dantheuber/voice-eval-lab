# Using voice-eval-lab

A walkthrough for someone testing the bot. You need a browser and, for mic calls, headphones. The bot under test is Ada, the support agent for Fernhill Internet, a made-up company that sells email hosting, website hosting, domains, and SSL.

Press `?` on any screen to see its keyboard shortcuts. `Ctrl K` jumps to anything.

## First run

1. Check the machine. This changes nothing, and it tells you what to fix.

   ```bash
   python3 scripts/doctor.py
   ```

2. Install the model servers and models, if they aren't there yet.

   ```bash
   scripts/setup.sh
   ```

   A clean machine downloads 24.9 GB and uses about 31 GB of disk, and the script asks first. Add `--check` to see what's present without changing anything. Run it again whenever you like, since it skips what it finds.

3. Start everything.

   ```bash
   ./run.sh
   ```

   It builds the dashboard if needed, starts the model servers, and prints two addresses:

   - `http://127.0.0.1:18080` on this machine.
   - `https://<this machine's IP>:18443` from another device on your network. That address has a self-signed certificate, so the browser warns once and you have to accept it. Browsers only allow the mic on localhost or HTTPS, so this is what makes mic calls possible from a second machine.

4. Wait for `All models are ready.` The dashboard also shows health for each role. The server log is `data/logs/serve.log`, and each model server has its own log next to it.

To stop, press Ctrl-C in that terminal. Model servers stop with it. Press Ctrl-C again if it takes too long.

Useful variations:

| Command | What it does |
| --- | --- |
| `./run.sh --fake` | No models. Rule-based bots and tone-coded audio, for a quick look at the dashboard. Its data stays in `data/fake`, away from real results |
| `./run.sh --lazy` | Starts model servers only when a call needs them |
| `./run.sh --profile local-fast` | Another stack. Run `scripts/setup.sh --profile local-fast` first. The same goes for picking it on the settings screen, which says what's missing and the command to run, and changes nothing until it's all there |
| `./run.sh --http-port 28080 --https-port 28443` | Other dashboard ports, when the defaults are taken |

`--profile` also saves the choice as the active profile. A plain `./run.sh` always starts `local-default`, even if you switched on the settings screen last time.

If `run.sh` says a port is taken, it tells you which process holds it. It won't start until that port is free.

`voicelab serve` run directly makes the same check for each model server. A health answer doesn't say which model a server has, so a server on the port is only used if it has the same command and directory, which is what a killed voicelab leaves behind. Anything else stays down, and `/settings` and `/api/health` show who holds the port.

## Run a scenario

A run is a batch of bot-to-bot calls. Each pairs a scenario (what the caller wants) with a persona (who is calling).

1. Open **Run launcher** (`g l`).
2. Tick cells in the grid. Each row is a scenario and each column a persona. Cells show the last 7 days' pass rate on the selected profile.
3. Set the options:
   - **Repetitions per pair.** Both sides are stochastic, so use 3 or more to get a pass rate you can read. 1 only shows whether the pair can pass at all.
   - **Model profile.**
   - **Call mode.** `live` runs in real time with measured timing. `turn-based` is slower than real time with each side waiting, and timing isn't measured. `auto` picks from the profile.
   - **Line conditions.** `per persona` uses each persona's own line. The others apply one condition to every call.
   - **Concurrency.** 2 live calls fit the GPU. Turn-based can go to 4. More than one two-sided call at once was not covered by the benchmarks. `local-realtime` runs one at a time, because its server holds one session. A mic call counts too: it goes ahead of queued bot calls, but during a bot call the Talk screen tells you to wait for that call to end.
4. Press **Launch** or `Ctrl Enter`.

The run page streams the transcript as it happens. **Pause after current calls** and **Stop** are at the top. Stopping marks the calls in flight as failed.

Evaluation starts by itself when each call ends. On 18 real calls it took 34 to 65 s, so expect a verdict about a minute after a call finishes.

A call ends as `pass`, `fail`, `needs_review`, or `invalid`. See [How evals work](../README.md#how-evals-work). An `invalid` call is a test problem: the caller never reached the scenario, so it says nothing about the bot and stays out of pass rates.

## Review a call with the keyboard

Press `g r` from anywhere to open the next call needing review. The review screen has the waveform, the transcript, and the findings list. You can do all of it without the mouse:

| Key | Action |
| --- | --- |
| `Space` | Play or pause |
| `j`, `k` | Next or previous finding |
| `Enter` | Play the selected finding's clip |
| `y` | Confirm the finding |
| `n` | Dismiss the finding |
| `c` | Add a note to the finding |
| `Left`, `Right` | Seek 2 s |
| `Shift Left`, `Shift Right` | Seek 10 s |
| `s` | Cycle playback speed |
| `m`, `b` | Mute the caller, mute the bot |
| `f` | Show findings only, or everything |
| `t` | Focus the transcript at the current turn |
| `1`, `2`, `3`, `4` | Set your verdict: pass, fail, needs review, invalid |
| `]`, `[` | Next or previous call needing review |

A normal pass over a call goes like this: `j` to a finding, `Enter` to hear it, `y` or `n`, repeat, a number key for your verdict, then `]`.

Your verdict sits beside the machine verdict and doesn't replace it. Advisory findings are listed so you can dismiss them, but they don't count as open.

Two things to keep in mind:

- On a turn-based call the gaps are synthetic. Timing checks are skipped, and the screen says so.
- Delivery and audio quality findings are advisory. Their limits were never checked against listener ratings, so trust your ears over them.

## Talk to the bot with headphones

1. Open **Talk to the bot** (`g t`), on this machine or through the HTTPS address. Allow the mic when the browser asks.
2. Put on headphones. On speakers the bot may hear itself. Echo handling exists, but it has only been tested on synthetic audio, not in a real room.
3. Under **Before you call**, pick the mic and speak. The level bars should reach the middle at a normal voice.
4. Choose **Record as**. `mic call` keeps the recording and evaluates it. `unlisted` deletes the call when it ends, with no evaluation.
5. Pick what you're calling about. Choosing a scenario gives you its fact sheet to read from. **Call freely** skips the scenario, and scenario rules can't be judged. You can also pick a persona and a line condition for your audio.
6. Press **Start call** or `Enter`. It stays disabled until the models are ready.

During the call:

- Talk over the bot whenever you like. It should stop, and it remembers only what you actually heard.
- `m` mutes you.
- `Esc` twice ends the call.

When the bot hangs up the screen moves on by itself, and closing the tab ends the call straight away. If a model fails mid-call, the screen says so and keeps what was recorded. The call is evaluated the same way as a bot call, and the screen links to the review when it's done. Mic calls always have real timing, so overlaps and barge-in reaction get measured. If the bot stops for noise or echo and your turn transcribes to nothing, the call logs a `false_barge_in`.

## Write a persona

A persona is who the caller bot pretends to be. Open **Personas**, then choose to create one, or open `/personas/new`.

1. **Who they are.** Name, age, temperament, and a one-line summary for lists.
2. **Speaking style.** This goes into the caller LLM's instructions word for word, so write instructions: "Short sentences. Trails off when confused." Not a biography.
3. **Scales.** Patience, verbosity, interruption tendency, and tech savviness, each 1 to 5.
4. **Voice.** Pick a TTS voice and use the preview line to hear it. The list comes from the active profile's caller TTS. Under `local-default` those are the 14 Pocket presets cleared for use. Under `local-fast` they're Phonon voices. A named voice belongs to one TTS, so a run on another profile is refused before it starts. The default, shown as "the profile's own voice", works on all of them.
5. **Line conditions.** Codec, background noise, SNR, packet loss, and one-way latency, applied to this persona's audio before the bot hears it. Lower SNR is noisier. Packet loss and line noise were not covered by the benchmarks.
6. **How this reads to the caller model** shows the exact instructions the caller LLM gets. Read it before saving.

Press **Save**. Seeded personas come from `seeds/personas/*.yaml` if you'd rather edit files.

## Write a scenario

A scenario is the issue a persona calls about. Open **Scenarios**, then choose to create one, or open `/scenarios/new`. The seeded `seeds/scenarios/double_charge_refund.yaml` is a good model to copy.

**Draft with AI** turns a plain description into a full draft for you to edit. It uses the judge LLM, and it marks what it guessed.

Fill in these parts, or check them after a draft:

1. **The issue.** A title and tags, a description for people, and what the caller wants in their own words. The caller bot sees that last field.
2. **Mock customer.** Each call gets a fresh copy of that customer's records.
3. **Turn limit.** The call ends as invalid if the caller hasn't reached its checkpoints by then.
4. **Caller's fact sheet.** The caller may only use these facts. Without a fixed sheet it invents details. Add everything the bot might ask for, such as billing zip and PIN.
5. **Beats, in order.** Each beat is what the caller should do next. A beat with a checkpoint says what proves it was reached. A missed required checkpoint makes the call invalid, not failed, unless the bot did something forbidden or ended the call itself. That's how a broken test stays separate from a bad bot.
6. **Rules the bot is judged on.** `must` and `must not` rules, each with a severity. The judge cites evidence for each one, a quote or a tool call. In a scenario's YAML, `needs_beat` on a rule or an expected change names the beat it depends on, so the bot only answers for it once the caller got that far. A rule with no link is excused by any missed required beat.
7. **Expected database changes.** The table, a filter, and whether a row was added, changed, or unchanged. These compare real database state, so they don't depend on the judge. An example is one row added to `refunds` with `invoice_id: FH-4419`.
8. **Pronunciation watch list.** Tokens the pronunciation check treats as critical, such as `FH-4419` and `$49.99`.

Then press **Dry run (text only)**. It plays the scenario as text with no audio and shows which checkpoints it reached. It's the fast way to catch a bad script before spending call time. Saving bumps the version, and past calls keep the snapshot they ran with.

## Upload a recording

You can evaluate a call recorded somewhere else. Open **Upload**.

1. Choose a `.wav`, `.flac`, `.mp3`, or `.ogg` file, then press **Upload and probe**. Files can be up to 2 GiB and 2 hours.
2. If the file is stereo, say which channel is the caller. You can listen to the first 10 seconds to check. A mono file gets split by energy, which is a guess and can put words on the wrong side. Reply latency, overlap, and barge-in checks are skipped, since they need to know who spoke.
3. Add what you know. Each field unlocks more checks:
   - **Scenario** lets the judge score its rules.
   - **Bot transcript with intended text** is one bot turn per line, or JSON `[{ "text": "...", "startS": 1.2 }]`. Without it, pronunciation can only compare two ASR passes.
   - **Tool log** is one JSON tool call per line. Without it, tool and database checks are skipped.
4. Press **Evaluate**.

Read the summary's "Not checked" line on the review screen. An uploaded call is the same artifact with less evidence, and that line lists what wasn't checked. An upload with no tool log always ends as `needs_review`, because it can't show what the bot did to the account. Its findings say what to listen to, and you set the verdict.

## Compare two runs

Use this to see what a change did, such as one batch on two model profiles or two bot versions.

1. Run the same scenarios and personas twice, and give each run a label.
2. Open **Compare**, then pick **Run A** and **Run B**. You can also press **Compare** on a finished run's page.

Read it in this order:

1. **Per-criterion rows.** Real regressions show up here. Headline pass rates from small runs mostly overlap.
2. **Measures.** Reply latency, critical mispronunciations, missing database changes, and so on. A measure that can't be compared says why. Latency, for one, is not comparable against a turn-based run.
3. **Cells.** Pass counts for each scenario and persona pair.

Scenarios in only one run are dropped from the comparison and listed. Criteria match by id, so renaming a rule's id makes it look new. Invalid calls are shown apart, because they say something about the test, not the bot.
