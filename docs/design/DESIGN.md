# voice-eval-lab dashboard design

Design direction, screen specs, tokens, and developer handoff for the dashboard. The prototype in `prototype/` is the reference for how things look and behave; this document says why, and fills in states the prototype does not show. Open `prototype/index.html` directly in a browser; nothing is fetched from the network.

## 1. Direction

The person using this currently tests a voice bot by phoning it and acting. The dashboard exists so that most of their day is judgment, not listening. The one number that matters is time from "this call has a finding" to "I have heard the 10 seconds that matter and decided". Everything else is arranged around that.

The point of view, in short:

- **It is a bench instrument, not a dashboard product.** Dense, monospace where the data is an identifier or a time, one accent colour, no hero numbers, no cards floating on gradients. A QA person stares at this for hours; it should look like a logic analyser, not a marketing page.
- **Findings are the unit of work, not calls.** The review screen opens on the first open finding with its clip cued. `y` or `n` decides it and moves to the next. When the last finding is decided, the call verdict bar is the next thing in focus. `]` goes to the next call. A whole call can be reviewed without touching the mouse or hearing more than the flagged clips.
- **Evidence sits next to the claim.** Every finding shows its quote, its intended vs heard text, or its database diff inline. The reviewer should never have to go looking for why the evaluator said what it said.
- **The UI tells the truth about what it knows.** Turn-based calls carry a dashed badge, their timing section says "not measured", and the gaps on the waveform are labelled synthetic. Advisory findings say "cannot fail the call" in their header. Pass rates always carry n and a 95% interval. Invalid is drawn as a hollow dashed badge, visually "not a result", because it is a test problem and not a bot failure.
- **Dark by default, light as a first-class theme.** Dark is kinder for long sessions with waveforms. Both themes use the same token names and both have been checked for contrast.

What was deliberately not done: no purple, no glassmorphism, no emoji, no big stat tiles, no charts that need a legend to be read, no modal-driven flows for the main loop.

## 2. Information architecture

Left rail, grouped by what the person is doing:

```
Results        Overview · Calls (badge: calls needing review) · Compare
Test design    Scenarios · Personas
Run            Run launcher · Talk to the bot · Upload recording
System         Models and settings
```

Call review is not in the rail; it is reached from any call reference (Calls list, Overview queue, run progress, compare drill-down, `g r`). The rail highlights Calls while reviewing.

The rail footer always shows the active model profile, model readiness ("6 of 7 models"), the LAN address, and the theme toggle. The topbar holds breadcrumbs, a jump box (`Ctrl K`: call id, scenario, run, persona), and page actions.

URL state: every filter, selected call, and selected finding is in the URL, so a reviewer can paste a link to a specific finding (`/calls/c_1030_0917_a3f2?finding=f4`).

## 3. Main flows

**Review loop (the one that matters).**
Overview "Needs a human" or `g r` → review screen opens on the first open finding, clip cued → `Enter` plays the clip (1 s before to 1 s after) → `y` confirm or `n` dismiss, optional `c` note → selection advances to the next open finding, clip cued → ... → last finding decided, toast says "set the call verdict" and focuses the verdict bar → `1`-`4` sets verdict → `]` next call. Counted in the prototype: a 6-finding call with no notes is 14 keystrokes (`Enter` and `y` or `n` per finding, one verdict key, `]`).

**Launch a run.** Run launcher → tick scenario × persona pairs (cells show last week's pass rate so weak pairs stand out) → reps, profile, mode, line → Launch → progress view with the current call streaming → calls become clickable as evals finish → Overview queue fills.

**Talk to the bot.** Setup (mic level, headphone advice, pick a scenario or persona or call freely) → in-call (your fact sheet and beats on the left, who is speaking, live transcript, interruption events, big End call) → "Call ended, evaluating" with each evaluator ticking over → review opens as results arrive.

**Build a scenario.** Describe it in plain words → Draft with AI → proposal shown as a diff against the form, facts the model guessed are tagged → Accept into new or this scenario → edit → Dry run (text only, no audio) → Save as a new version.

**Compare.** Pick two runs with the same scenario set → headline pass rates with intervals and a sample-size warning → per-criterion table where regressions actually show → scenario × persona cells → drill to the failing calls.

## 4. Screen specs

Each screen lists: layout, what the person does, states, and the keys that work there. Global keys are in section 5.

### 4.1 Overview (`index.html`)

Layout: a warning strip if a model is not ready; three stat panels plus the review queue; recent runs table; three panels: pass rate by scenario, models, most common open findings.

Behaviour:
- Pass rate panel shows pass of *valid* calls, the 95% Wilson interval, the stacked bar, and a legend with counts. Invalid is listed separately and never in the denominator.
- "Needs a human" shows calls, open findings, the oldest item, and a time estimate from the reviewer's own recent pace (median time per finding over the last 50 decisions × open findings).
- "Ran recently" shows a 24-hour sparkline of calls per hour. It is a glance, not a chart to read; no axis.
- Queue items open review directly. Order: oldest first, so nothing rots.
- Recent runs: one row per run with mode badge, outcome bar, pass % and interval. Row click goes to the call list filtered to that run.
- Pass rate by scenario: rows, not a bar chart with axes. Each row has the stacked bar, pass %, interval, and n. Low-n rows are not hidden, their interval is simply wide.

States:
- Loading: skeleton rows in each panel, numbers show as `--`.
- Empty (fresh install): the stat panels say "No calls yet", the queue says "Nothing to review", and the recent runs panel has a single "Launch a run" button plus a "Talk to the bot" link. Pass rate by scenario lists the seeded scenarios with "no data".
- Model not ready: the warning strip (shown in the prototype) names the model, the elapsed time, and the usual time, and says what still works ("calls can run, evals queue").
- Error: the panel that failed shows the request that failed and a retry button; the rest of the page still renders.

### 4.2 Calls (`calls.html`)

Layout: title and count, filter row, dense table.

Filters: verdict as a segmented control with counts, scenario, persona, mode, model profile as selects, run as a removable chip when arrived at from a run. "Open findings only" checkbox. All filters are URL parameters.

Columns: call id (mono), verdict (with the invalid reason inline under the badge), scenario, persona ("you" for mic calls, "unknown" for uploads), mode, profile, run and rep, duration, reply p50, overlaps, findings with an "n open" tag. Turn-based rows show `n/a` with a tooltip in the timing columns rather than a dash, so it reads as "not measured" and not "zero".

Keys: `j`/`k` move the selection marker, `Enter` or `o` opens the call, `Esc` clears filters.

States: loading skeleton; empty state names the active filters and offers "launch a run for this scenario"; a run still in progress shows running and queued rows with the running verdict badge.

### 4.3 Call review (`review.html`, `review.js`)

This is the screen that pays for the rest. Three regions in a fixed-height layout: header, timeline across the top, transcript left, findings and evaluators right.

**Header.** Call id, verdict, mode badge, scenario, persona, profile, run and rep, call time (and wall clock time for turn-based), turn count against the limit. Right side: "finding i of n", prev and next call needing review. Turn-based calls get a one-line note under the header explaining what is and is not real. It is shown once per call, not repeated anywhere else on the screen.

**Timeline.** Toolbar: play/pause with the key shown, time readout, speed (1x, 1.5x, 2x), current clip label, legend, and either the p50 reply latency (live) or a "gaps synthetic" tag (turn-based). Below, five lanes with a 64 px label column:

1. Findings lane: numbered pins at each finding's timestamp, coloured and bordered by severity (high solid red border, medium amber, low grey, advisory dashed grey). Decided findings fade. Click plays the clip.
2. Caller waveform (blue).
3. Bot waveform (orange). The two waveforms are stacked and share one time axis so an overlap is visibly two bursts in the same column; overlap intervals also get a red diagonal hatch across both lanes.
4. Events lane: tool calls (wrench), database changes (cylinder, accent colour), barge-ins (red). Hover for the full call and result; click seeks to half a second before.
5. Ruler.

A clip range is drawn as a translucent accent band with solid edges; the playhead is a 1 px line with a flag. Click anywhere on a waveform to seek and clear the clip.

**Transcript.** Each turn: speaker label in the channel colour, timestamp, text. Bot turns show the text the ASR heard; when it differs from the intended text, the intended token is struck through in red and the heard token is highlighted amber, with the full intended line underneath. Turns with findings carry a small numbered tag matching the pin. Interrupted bot turns say "interrupted"; overlapping caller turns say "overlaps bot speech". Tool and database events sit inline as mono rows. The active turn follows the playhead and scrolls into view while playing. Filter: all turns, with findings (`f`), tools and db.

**Findings panel.** Count of open findings and the key hints. Each card: severity, time, source evaluator; title; evidence (quote in italics, or intended vs heard in mono, or the database diff); a clip button with its range; confirm (`y`), dismiss (`n`), note (`c`). Confirmed and dismissed are written next to the time in words, not only by fading. Advisory findings say "advisory, cannot fail the call" in the header and are drawn with a dashed top border.

Decision behaviour: confirming or dismissing toggles (press again to reopen), updates the open count and pin, shows a toast, and advances to the next open finding with its clip cued. Notes save on blur or `Ctrl Enter`, `Esc` cancels. A saved note renders as a quiet block with an edit link.

**Below findings:** rubric table (one row per criterion with pass/fail/needs-review badge, "must" or "must not" label, the quoted evidence with a timestamp link), database changes as an expected/actual diff with missing rows in red and allowed extras in green, timing (or "not measured (turn-based)" in each cell), and the evaluator list with status, one-line summary, and model used. The audio listener is labelled advisory here too.

**Verdict bar.** Sticky at the bottom of the panel: current machine verdict and a segmented control for the human verdict with `1`-`4`. Setting it toasts and points at `]`.

Keys on this screen:

| Key | Action |
| --- | --- |
| `Space` | play / pause |
| `Enter` | play the selected finding's clip |
| `j` / `k` | next / previous finding (cues its clip) |
| `y` / `n` | confirm / dismiss selected finding |
| `c` | add or edit a note on the selected finding |
| `←` / `→`, with `Shift` | seek 2 s / 10 s |
| `f` | toggle "with findings" transcript filter |
| `s` | cycle speed |
| `t` | focus the active transcript turn |
| `1` `2` `3` `4` | human verdict pass / fail / needs review / invalid |
| `]` / `[` | next / previous call needing review |

States:
- Evals still running (opened from Talk or a live run): findings panel shows each evaluator with a running badge and the findings list grows as results arrive. Timeline and transcript are usable immediately because the recording exists.
- Turn-based: header note, "gaps synthetic" tag, timing section not measured, no overlap hatch, timing evaluator "skipped". Shown in the prototype at `review.html?call=c_1030_0802_1c3d`.
- Upload with reduced evidence: the evaluator list says which checks were skipped and why ("no tool log"), the rubric section is replaced by "no scenario selected" or shows judge-only results, database changes says "not available", transcript shows heard text only with no intended line. The verdict bar still works; the machine verdict is always needs review for these.
- Invalid: the header badge is invalid and a note gives the caller-validity reason and the checkpoint reached. Findings about the bot are still listed but the verdict bar says "invalid runs do not count against the bot" and the pass/fail buttons are disabled unless the reviewer first un-marks invalid.
- Audio missing or corrupt: waveform lanes show "recording unavailable" with the file path; transcript and findings still render; clip buttons are disabled.
- Narrow (under 900 px): the sidebar collapses under the transcript with a max height, the rail collapses to icons, the legend text hides. Still fully keyboard operable.

### 4.4 Persona builder (`persona.html`)

Layout: form left, "how this reads to the caller model" and recent calls right.

Fields: name, age, temperament (select), speaking style (free text, goes to the caller LLM verbatim), four 1 to 5 scales with end labels (patience, verbosity, interruption tendency, speaking rate), voice (select of TTS voices across installed TTS providers), speed, preview line with a Preview button (renders through the active TTS with the line condition applied; status line shows "rendering" then "played 3.4 s, road noise at -22 dBFS, mu-law"), line conditions (codec, background noise type, level, packet loss).

The rendered instruction block updates live so the author sees exactly what the model will be told. An info note explains what interruption 5 means in live vs turn-based mode.

States: preview while TTS is loading shows the model-loading health dot and the elapsed time; preview error shows the server message inline and keeps the form editable.

### 4.5 Scenario builder (`scenario.html`)

Layout: form left (the issue, fact sheet, beats, rules, expected database changes, pronunciation watch list), AI draft panel right, sticky.

- The issue: name, tags, the caller's complaint in their own words, mock account (select from seeded accounts with a "view seed" link), turn limit with the invalid rule spelled out.
- Fact sheet: key and value pairs. The help text says the caller may only use these facts.
- Beats: ordered, draggable, each with a checkpoint toggle. Checkpoints decide validity, not bot pass.
- Rules: must (C1..) and must not (N1..) with distinct coloured keys. Help text says a failed must-not always fails the call.
- Expected database changes: a small textual DSL (`table[key].field: from -> to`, `table: +1 row where ...`, `table: any (allowed)`).
- Pronunciation watch list: comma separated; numbers and names from the fact sheet are added automatically.
- Versions: saving creates a new version; old calls keep theirs. Shown in the subtitle.

Draft with AI: plain description → Draft button → skeleton with "usually 5 to 10 s" → proposal shown as green insert blocks per section, facts the model guessed carry a "guessed" tag → Accept into a new scenario, accept into this one, or discard. "Replace empty fields only" is on by default so a draft does not overwrite edits. Nothing is saved until accepted.

States: draft failure shows the error and keeps the description; an account that no longer exists in the seed is flagged on the select; the Save button is disabled with a reason if there is no checkpoint beat.

### 4.6 Run launcher and progress (`run.html`, `run.html?view=progress`)

Launcher: scenario × persona matrix with row and column selectors. Each cell shows last week's pass rate for that pair on the chosen profile, amber under 70%, so the matrix doubles as a map of where the bot is weak. Right column: repetitions (1, 2, 3, 5, 10, with the stochastic warning), model profile (unavailable profiles disabled with the reason), call mode (auto picks live or turn-based from the latest benchmark and says which and why), line conditions, concurrency. Summary: call count, effective mode badge, time estimate (wall clock for turn-based), eval readiness. Launch button names the number of calls.

Progress: header with running badge and mode; left, a segmented progress bar with one cell per call coloured by outcome (running cells pulse), a legend with counts, the pass rate so far with the interval and a note not to read it yet; the call list with verdict glyphs and open links as calls finish. Right, the live call: scenario and persona tags, clock, transcript streaming in with a caret, tool and barge-in events inline, a footer naming the other concurrent call with "watch instead". Actions: pause after current calls, stop run.

States: queued (no models free) shows position in queue; a failed call (transport error) shows as fail with the reason and does not count as a bot fail unless confirmed; a run stopped early keeps partial results and says so in the run row.

### 4.7 Talk to the bot (`talk.html`, `?phase=setup|incall|evaluating`)

Setup: mic select, live level meter with guidance, headphone note that explains what happens on speakers, record-as segmented control, scenario or call freely, optional persona, line condition applied to your own audio. Start call is `Enter`.

In call: left, your sheet (who you are, the problem, beats with completed ones struck through, must-not-say); events log with interruptions in red. Centre, two speaker tiles, "you" and "bot", each with a state word (speaking, listening, quiet), a level meter in the channel colour with hot segments in red, and a line of context (echo cancellation on, headphones detected; interrupted once, queued audio dropped at 00:58). The tile for whoever is speaking gets a coloured border. Below, the live transcript with a caret on the partial turn. Footer: clock, a note that the live transcript is from the bot's STT and may lag, mute (`m`), and a large red End call that needs `Esc` twice (the first press toasts).

Evaluating: a single panel listing each evaluator with done, running, or queued badges and a one-line result as it arrives, "open review as results arrive", "call again", and a note that leaving is fine.

States: mic permission denied shows the browser instruction and the LAN/HTTPS reminder; no headphones detected keeps the warning visible in the "you" tile; bot model loading shows the setup Start button disabled with "bot LLM loading, 18 s"; WebRTC disconnect mid-call shows a red strip and keeps the recording so far.

### 4.8 Upload a recording (`upload.html`)

Drop zone with format and channel guidance; after selection, a file row with duration, channels, sample rate, size, and the channel detection result. Optional context: scenario, bot version, intended bot transcript, tool log. Right, the evidence panel lists each kind of evidence with a check or cross and what it enables or skips, and a warning that with reduced evidence the call cannot be marked pass or fail automatically. Evaluate button with a time estimate.

States: mono file shows the diarisation warning and lets the user pick which speaker is the bot after a 10 s preview; unsupported file shows the error inline; a scenario chosen later re-runs only the judge.

### 4.9 Models and settings (`settings.html`)

Active profile select with the YAML path and the latest benchmark result that decides live vs turn-based. Roles table: role, interface name (LLM, STT, TTS, RealtimeVoice, AudioAnalyst, AudioQuality), adapter select, status with the health dot and detail, licence. Any Bedrock adapter shows an "untested against live AWS" tag next to the select, and the Swap a provider panel repeats the warning with region and credential status and a disabled Test connection. Evaluation thresholds. Right column: model servers with state and detail, a loading note with elapsed and usual time, GPU memory, access URLs.

States: a server down shows red with the last error line and a restart button for that server only; restarting shows loading; changing an adapter shows "restarting role" inline and blocks only that role.

### 4.10 Compare (`compare.html`)

Two run selectors. A note about sample size with both intervals. Two headline panels with pass rate, n, interval, mode badge, and the outcome bar. Per rubric criterion: pass counts and percentages for each run and a delta column with up/down glyphs plus colour (green better, red worse). A callout names the largest regression and links to those calls. Other measures: timing rows are marked "not comparable" when either run is turn-based; invalid runs are listed as "test, not bot"; advisory counts are listed but not scored. Scenario × persona cells show A → B pass of n reps with a glyph.

States: runs with different scenario sets show only the intersection and say what was dropped; a run still in progress is comparable but marked partial.

## 5. Keyboard map

Global (any screen, not while typing in a field):

| Key | Action |
| --- | --- |
| `?` | keyboard help |
| `Ctrl K` | jump to call, scenario, run, persona |
| `g o` | Overview |
| `g c` | Calls |
| `g r` | next call needing review |
| `g t` | Talk to the bot |
| `g s` | Scenarios |
| `g l` | Run launcher |
| `Esc` | close dialog or clear filters |

Screen keys are listed with each screen above. Rules: single letters for the review loop, no modifier keys for anything done hundreds of times a day, destructive actions need two presses (`Esc Esc` to end a call) or a modifier (`Ctrl Enter` to launch). Every key is printed on the control it triggers, as a `kbd` chip.

## 6. Component inventory

Shell: rail (brand, grouped nav items with optional count, footer with profile, health, theme), topbar (breadcrumbs, jump box, actions), toast, keyboard help dialog.

Primitives: button (default, primary, danger, danger solid, ghost; sm, lg), kbd chip, tag, verdict badge, mode badge, severity label, health dot, note (plain, info, warn, err), panel (header, body, flush variant), table (sticky header, selectable rows, mono id and time cells), field with label and help, input, select, textarea, checkbox, segmented control, 1 to 5 scale, chip, empty state, skeleton.

Data display: stat panel, stacked rate bar with legend, sparkbar, rate row, confidence interval text, progress bar (segmented per call).

Review: timeline (toolbar, lanes, waveform canvas, pins, event markers, clip band, playhead, ruler), transcript turn (caller, bot, event; diff spans; finding tag; barge and overlap notes), finding card, rubric table, database diff block, key-value block, verdict bar.

Builders: matrix table, beat row, rule row, fact row, AI draft panel with diff blocks, voice preview control.

Talk: speaker tile, level meter, live turn list with caret, events log, end bar.

## 7. Design tokens

Both themes define the same names. Values below are the prototype's (`prototype/styles.css`).

### Colour

| Token | Dark | Light | Use |
| --- | --- | --- | --- |
| `--bg-0` | `#0e1013` | `#eef0f3` | app background |
| `--bg-1` | `#15181d` | `#ffffff` | panels, rail, topbar |
| `--bg-2` | `#1b1f26` | `#f6f7f9` | inputs, hover rows |
| `--bg-3` | `#232832` | `#eceef2` | hover, kbd, skeleton |
| `--bg-4` | `#2b313c` | `#e1e4ea` | pressed, meter track |
| `--line-1` | `#262c35` | `#d8dce3` | dividers |
| `--line-2` | `#343c48` | `#c3c9d3` | control borders |
| `--ink-1` | `#e4e8ee` | `#15181d` | primary text |
| `--ink-2` | `#a3acb9` | `#4b5563` | secondary text |
| `--ink-3` | `#7a8491` | `#697585` | tertiary, labels |
| `--ink-4` | `#4b5460` | `#a0a8b4` | disabled, hairlines |
| `--accent` | `#4c9be8` | `#1f6fd1` | interactive, selection, db events |
| `--accent-soft` | 14% accent | 10% accent | selected backgrounds |
| `--focus` | `#8cc2ff` | `#1f6fd1` | focus ring |
| `--pass` | `#3fb950` | `#167a33` | verdict pass, done |
| `--fail` | `#f46a6e` | `#c9262c` | verdict fail, high severity, overlap, barge-in |
| `--review` | `#e0a62c` | `#875a00` | needs review, medium severity, loading |
| `--invalid` | `#8b95a3` | `#5f6b7a` | invalid (always with dashed border) |
| `--caller` | `#3987e5` | `#2a78d6` | caller channel fills |
| `--bot` | `#e0702a` | `#d4571f` | bot channel fills |
| `--caller-text` | `#3987e5` | `#2263b5` | caller speaker labels |
| `--bot-text` | `#e0702a` | `#b4470f` | bot speaker labels |
| `--overlap` | 28% fail | 22% fail | hatch across both lanes |

Each verdict and severity colour also has a `-soft` background at 10 to 16% for badges. Caller and bot were checked as a pair with the dataviz palette validator in both modes: CVD ΔE 26 and normal ΔE 32, both over the floors, and both over 3:1 against their surfaces. Verdict colours are status colours and are never used for series.

### Type

System stacks so the prototype has no font dependency. Production may self-host Inter and JetBrains Mono under `web/public/fonts`; never load from a CDN.

| Token | Value |
| --- | --- |
| `--font-ui` | `system-ui, -apple-system, "Segoe UI", Inter, Roboto, sans-serif` |
| `--font-mono` | `ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace` |
| `--fs-0` .. `--fs-6` | 11, 12, 13, 14, 16, 20, 26 px |
| body | 13 px / 1.4 |
| tables, cards | 12 px |
| section labels | 11 px, uppercase, 0.04 em tracking, `--ink-2` |
| mono | 0.93 em of context, tabular numerals for times and counts |

### Spacing, radius, motion

- Spacing scale: 4, 8, 12, 16, 24, 32 px. Panel padding 12, page padding 16, table cell 6 × 10.
- Radius: 3 px controls and badges, 5 px buttons and inputs, 8 px panels and dialogs.
- Motion: 100 ms for hover and press, 160 ms for panel and toast, ease `cubic-bezier(.2,.7,.2,1)`. Playhead moves on animation frames with no easing. Running badges spin at 900 ms, loading dots pulse at 1.2 s. `prefers-reduced-motion` collapses all of it to near zero.
- Layout: rail 180 px (52 px under 900 px), topbar 44 px, review sidebar 380 px (330 under 1180 px).

## 8. Accessibility

- Nothing is colour alone. Verdicts have a word and a distinct glyph (dot, square, triangle, hollow dashed ring). Severity is a word. Caller and bot lanes are labelled. Overlap is a hatch pattern and a word in the transcript. Deltas have arrows. Health states have words next to the dot. Mode badges have an icon, a word, and turn-based adds a dashed border and "slow".
- Contrast, computed with the WCAG formula against `--bg-1`: `--ink-1` 14.5:1 dark and 17.8:1 light; `--ink-2` 7.8:1 and 7.6:1; `--ink-3` 4.7:1 and 4.7:1 (used for labels and detail only). Verdict text on its own soft badge background: pass 5.6 dark and 4.7 light, fail 5.0 and 4.7, review 6.4 and 5.1. Channel fills are graphics and sit at 4.9 and 5.5 dark, 4.4 and 4.1 light, all over the 3:1 graphics floor; speaker labels use the `-text` variants, 5.5 and 5.6 in light. Invalid text 5.9 and 5.4. Accent on panel 6.1 and 4.9. The script used is three lines of relative-luminance maths; re-run it if a token changes.
- Keyboard: everything in section 5 plus native tab order. Focus ring is 2 px in `--focus` with offset. Pins, event markers, transcript turns, and finding cards are buttons or focusable with `aria-pressed` for selection. Segmented controls use `aria-pressed`; the 1 to 5 scales are radio groups.
- Screen readers: rate bars carry `role="img"` with a full text label; waveforms are presentational and the transcript is the accessible equivalent; toasts should be an `aria-live="polite"` region in production.
- Motion and audio: no autoplay; clip playback needs a keypress or click. Reduced motion respected.
- Targets: buttons are at least 24 px tall at the small size and 28 px by default. Pins are 18 px with a larger hit area in production (the prototype uses the visible size).

## 9. Developer handoff

### Stack

The project is React, TypeScript, Vite, pnpm under `web/`. Recommendations:

- **Styling: plain CSS with custom properties, scoped by CSS modules.** The prototype's `styles.css` already is the token sheet; keep it as `web/src/styles/tokens.css` and `base.css`, and move screen styles into `*.module.css`. No Tailwind: the density here comes from a small number of hand-tuned components, and a utility layer would mostly be restating the tokens. No component kit skin (MUI, Mantine) because every one of them fights the density and the look.
- **Headless primitives for the hard accessibility: Radix UI** (`@radix-ui/react-dialog`, `-select`, `-tooltip`, `-popover`, `-toggle-group`). Unstyled, keyboard correct, small. Style them with the tokens.
- **Data: TanStack Query** for fetching and cache, with WebSocket subscriptions invalidating or patching queries (run progress, eval progress, live transcript). **TanStack Table** for the calls list (sorting, column sizing, virtualisation via `@tanstack/react-virtual` once calls pass a few thousand).
- **Routing: React Router**, with filters and selection in search params. Route list: `/`, `/calls`, `/calls/:id`, `/compare`, `/scenarios`, `/scenarios/:id`, `/personas`, `/personas/:id`, `/runs/new`, `/runs/:id`, `/talk`, `/upload`, `/settings`.
- **Keyboard: a single `useHotkeys` hook** (write it, or `react-hotkeys-hook`) with a scope stack so screen keys do not fire inside inputs or dialogs. The `g` chord is a 900 ms pending state.
- **Audio: a plain `<audio>` element** for the stereo recording, with `currentTime` seeking and `timeupdate` driving the playhead. Clip play is `currentTime = start` then pause at `end` on `timeupdate`. For per-channel mute (listen to the bot only), route through Web Audio `ChannelSplitterNode` and two `GainNode`s. **Waveform: draw it yourself on canvas from server-provided peaks**, as the prototype does in about 40 lines. wavesurfer.js was considered and rejected: dual stacked channels, overlap hatching, pins, and event lanes are all custom drawing anyway, and its plugin surface adds weight without covering them.
- **Live call: Pipecat's small WebRTC transport** from the browser, with a data channel for transcript and events. Mic level from an `AnalyserNode` on the local stream. Echo cancellation via `getUserMedia` constraints (`echoCancellation: true, noiseSuppression: true, autoGainControl: false`).
- **State:** server state in Query; per-screen UI state in the URL; review session state (selected finding, playhead, clip, pending note) in a small zustand store so the timeline, transcript, and findings panel stay in sync without prop drilling.
- **Theme:** `data-theme` on `<html>`, persisted in `localStorage`, default dark, with `prefers-color-scheme` as the initial value on first visit.
- **Testing:** Vitest and Testing Library for components; Playwright for the review loop, including the full keyboard path (open, `Enter`, `y`, `y`, `n`, `1`, `]`).

### Data each screen needs

The API should be designed from these. Field names are suggestions; shapes matter more than names. Times are seconds from call start as floats; wall times are ISO strings.

**Overview** `GET /api/overview?profile=&window=24h`
`{ passRate: { pass, fail, review, invalid, valid }, queue: [{ callId, scenarioId, personaName, openFindings, why, createdAt }], reviewPaceSecPerFinding, callsPerHour: number[24], recentRuns: Run[], scenarioRates: [{ scenarioId, pass, fail, review, invalid }], health: Health[], topOpenFindings: [{ title, source, count }] }`

**Calls** `GET /api/calls?verdict=&scenario=&persona=&mode=&profile=&run=&openOnly=&cursor=`
`{ items: [{ id, runId, rep, scenarioId, personaId, mode: "live"|"turn"|"human"|"upload", profile, verdict, humanVerdict?, invalidReason?, findings, openFindings, durationSec, replyP50Sec?, overlaps?, createdAt }], next }`
Also `GET /api/review-queue/next?after=:id` for `]` and `g r`.

**Call review** `GET /api/calls/:id` returns one artifact:
```
{
  id, runId, rep, repOf, scenario: { id, name, version }, persona: { id, name, line }, mode, profile,
  verdict, humanVerdict, invalidReason, durationSec, wallClockSec, recordedAt, turns, turnLimit,
  audio: { stereoUrl, callerUrl, botUrl, sampleRate, peaks: { caller: number[], bot: number[], perSecond } },
  speech: { caller: [[start,end]], bot: [[start,end]] }, overlaps: [[start,end]],
  events: [{ t, kind: "tool"|"db"|"barge"|"hangup", label, detail, args?, result? }],
  transcript: [{ who: "caller"|"bot", t, end, heard, intended?, confidence?, words?: [{ w, t, end }], bargeIn?, overlap?, findingIds: [] }],
  findings: [{ id, t, end, severity: "high"|"med"|"low"|"advisory", source, title, evidence, intended?, heard?, quote?, clipUrl, state: null|"confirmed"|"dismissed", note?, decidedAt?, decidedBy? }],
  rubric: [{ id, kind: "must"|"must_not", text, result: "pass"|"fail"|"review"|"skipped", quote?, at?, note? }],
  dbDiff: [{ kind: "expected_present"|"expected_missing"|"unexpected"|"allowed", text }],
  timing: null | { p50, p95, overlaps, deadAirEvents, bargeStopSec },
  evaluators: [{ name, status: "queued"|"running"|"done"|"skipped"|"error", summary, model?, advisory?, skippedReason? }],
  evidence: { audio, transcript, intendedText, toolLog, dbSnapshots, scenario },
  next: callId | null, prev: callId | null
}
```
Writes: `POST /api/calls/:id/findings/:fid` `{ state, note }`; `POST /api/calls/:id/verdict` `{ verdict, note }`. Both return the updated call summary. `GET /api/calls/:id/events` is a WebSocket or SSE stream used while evaluators are still running: `{ type: "finding"|"evaluator"|"verdict", payload }`.

Clip audio: either `GET /api/calls/:id/clip?start=&end=` returning a WAV, or the client seeks the full recording. The prototype assumes seeking; clips as files are only needed for export.

**Personas** `GET/POST/PUT /api/personas[/:id]` with `{ id, name, age, temperament, style, patience, verbosity, interruption, rate, voice: { provider, voiceId, speed }, line: { codec, noise, noiseDb, packetLoss }, usage: { calls, passRate } }`. `POST /api/tts/preview` `{ text, voice, line }` → audio. `GET /api/voices` lists voices per installed TTS provider.

**Scenarios** `GET/POST/PUT /api/scenarios[/:id]` with `{ id, version, name, tags, issue, accountId, turnLimit, facts: [{ k, v }], beats: [{ text, checkpoint }], must: [{ id, text }], mustNot: [{ id, text }], expectedDb: string, watchList: string[], usage }`. `GET /api/accounts` for the seed list and `GET /api/accounts/:id/seed`. `POST /api/scenarios/draft` `{ description, base?: scenarioId }` → the same shape plus `guessed: string[]` naming fields the model inferred. `POST /api/scenarios/:id/dry-run` → a text-only transcript.

**Run launcher** `GET /api/matrix?profile=&window=7d` → `[{ scenarioId, personaId, pass, n }]`. `GET /api/profiles` → `[{ id, label, desc, health, liveCapable, benchmark: { p50, at } }]`. `POST /api/runs` `{ pairs: [{ scenarioId, personaId }], reps, profile, mode: "auto"|"live"|"turn", line, concurrency }` → `{ runId, calls, effectiveMode, estimateSec }`.

**Run progress** `GET /api/runs/:id` → `{ id, status, mode, profile, startedAt, calls: [{ id, scenarioId, personaId, rep, status, verdict? }], counts, etaSec }`. WebSocket `/ws/runs/:id`: `{ type: "call_started"|"turn"|"partial"|"event"|"call_done"|"run_done", callId, payload }` where `turn` and `partial` carry `{ who, text, t }`. `POST /api/runs/:id/pause`, `/stop`.

**Talk** `GET /api/devices` is browser-side. `POST /api/talk/sessions` `{ scenarioId?, personaId?, line?, record }` → `{ callId, webrtc: { offerUrl, iceServers } }`. Data channel messages: `{ type: "turn"|"partial"|"speaking"|"event"|"metrics", payload }` where `speaking` is `{ who, on }` and `metrics` carries bot reply p50. `POST /api/talk/sessions/:id/end` → the call id, then the review event stream.

**Upload** `POST /api/uploads` multipart with `{ file, scenarioId?, botVersion?, intendedTranscript?, toolLog? }` → `{ callId, probe: { durationSec, channels, sampleRate, channelMap }, evidence }`. Mono files get `POST /api/uploads/:id/channel-map` after preview.

**Settings** `GET /api/profiles/active`, `PUT /api/profiles/active { id }`. `GET /api/roles` → `[{ role, interface, adapter, options: [{ id, label, untested }], status, detail, licence }]`. `PUT /api/roles/:role { adapter }`. `GET /api/health` → `Health[]` with `{ name, model, state: "ok"|"loading"|"down"|"off", detail, elapsedSec?, usualSec? }`. `GET /api/system` → GPU, memory, URLs. `GET/PUT /api/thresholds`.

**Compare** `GET /api/compare?a=runId&b=runId` → `{ a: RunSummary, b: RunSummary, intersection: scenarioIds, dropped, criteria: [{ id, text, a: [k, n], b: [k, n] }], measures: [{ key, a, b, comparable, reason? }], cells: [{ scenarioId, personaId, a: [k, n], b: [k, n] }], worst: { criterionId, deltaPts, callIds } }`.

### Decisions the backend should know about

1. **Invalid is excluded from every denominator** the UI shows. The API should return invalid separately and never fold it into fail.
2. **Pass rates ship with n**; the UI computes the Wilson interval client-side, so the API returns counts, not percentages.
3. **Timing fields are nullable**, and null means "not measured" (turn-based, or upload without timestamps). The UI renders null as `n/a` with a reason. Do not return zeros.
4. **Advisory findings are a severity, not a flag**, and the verdict engine must ignore them when computing pass/fail. The UI shows them but never counts them in the "open" badge for the call list (it does count them on the review screen so they can be dismissed).
5. **Findings carry their own clip range** (`t`, `end`). The UI pads by 1 s each side for playback; the API should not pre-pad.
6. **Every call carries `mode` and `evidence`**; the review screen is driven entirely from those two fields, so uploads and mic calls need no special endpoints.
7. **The transcript is one ordered list** of caller turns, bot turns, and events, each with a time, so the UI can render it as a single scrollable timeline. Bot turns carry both `intended` and `heard`; the pronunciation evaluator's alignment (`words`) is optional but makes the diff exact.
8. **Human decisions are stored per finding and per call** with who and when, because they feed judge calibration. The API should accept a note with either.
9. **Live streams send partials**, so the UI can show a caret while a turn is still being recognised. A final `turn` message replaces the partial.
10. **Bedrock adapters return `untested: true`** in their option metadata; the UI puts the label next to every use of them, including on calls and in Compare.

### Prototype file map

```
docs/design/prototype/
  styles.css     tokens, primitives, screen layouts, responsive rules
  app.js         rail, topbar, theme, keyboard help, helpers (verdict, mode, rate bar, interval)
  data.js        Fernhill Internet sample data: personas, scenarios, runs, calls, two full review payloads
  review.js      call review behaviour: waveform drawing, simulated playback, findings loop, keys
  index.html     Overview
  calls.html     Calls list
  review.html    Call review (?call=c_1030_0917_a3f2 live, ?call=c_1030_0802_1c3d turn-based)
  run.html       Run launcher (?view=progress for the running view)
  talk.html      Talk to the bot (?phase=setup | incall | evaluating)
  scenario.html  Scenario builder (?draft=1 shows the AI proposal)
  persona.html   Persona builder
  upload.html    Upload a recording
  settings.html  Models and settings
  compare.html   Compare two runs
```

Any page accepts `?theme=light`. The review screen's playback is simulated (no audio file in the repo); the playhead, clip range, transcript sync, and all keys behave as specified.
