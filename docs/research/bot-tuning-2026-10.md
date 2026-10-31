# Tuning the two bots on real calls

Measured on 2026-10-31 on `local-default`. This records why the first 21 real calls went wrong, what changed, and what the changes did. Before and after numbers are in the [README](../../README.md#bot-tuning-results).

## Why the first 21 calls failed

Every transcript, tool log, and eval result was read. A call can have more than one cause, and the first listed is the one that decided it.

| Call | Scenario | Persona or profile | Verdict | Root cause |
| --- | --- | --- | --- | --- |
| `call_53e4a3164dc9` | forgotten_password | sysadmin | fail | Bot prompt: reset the password before verifying, in the same turn as the lookup. Said the wrong destination, and invented "the link doesn't expire". Caller drifted to MX records |
| `call_b8a6466506c8` | forgotten_password | chatty | fail | LLM tool mistake: made up a zip and PIN, then read the misheard address back. Caller thanked the bot and left before accepting the reset, so the bot ended the call |
| `call_f49ba608cf63` | locked_mailbox | sysadmin | invalid | STT heard "hart" as "heart". The bot read both back as the same sound, the caller argued from its sheet, and the mailbox was unlocked six times |
| `call_362375aa9277` | locked_mailbox | chatty | pass | |
| `call_8a3c0d1ea1c4` | expired_ssl | sysadmin | pass | The bot renewed before the yes, and no rule in the scenario catches that |
| `call_85691c115f1b` | expired_ssl | chatty | fail | Bot prompt: renewed before a yes, never said the date or that it was free, then checked DNS. Invented a `.com` domain |
| `call_fac103160741` | dns_mx_fix | chatty | invalid | Caller behavior: left before asking how long DNS takes. The bot read "win auto" back as a correction |
| `call_8fdf87441e4a` | double_charge_refund | mic | fail | LLM tool mistake: made up a phone, read invoices before verifying, and refunded the same payment twice |
| `call_90e10193e230` | domain_renewal | sysadmin | invalid | Bot prompt: charged the card with no price and no yes. The caller never got to approve, so the verdict excused a real bot failure. Eval weakness, see below |
| `call_9f9cc0ce6042` | domain_renewal | chatty | fail | Bot prompt: same charge with no yes, then ended the call |
| `call_5c9a477ca40a` | double_charge_refund | sysadmin | fail | Bot prompt: read the invoice before verifying, and refunded before a yes. LLM tool mistake: made up a zip and PIN |
| `call_e0e2273c2507` | double_charge_refund | chatty | needs_review | Bot prompt: refunded before a yes. Eval: a beat came back undecided because a judge quote wasn't found |
| `call_a882094acac7` | dns_mx_fix | sysadmin | fail | Bot prompt: left out "within an hour". The model looped on one tool and copied the hold line six times in one turn. Caller went off script on `dig` and TTLs |
| `call_2afa39f4a980` | already_refunded | bookkeeper | fail | Bot prompt: gave the refund status before verifying |
| `call_48cc57bf740c` | double_charge_refund | mic | fail | LLM tool mistake: refunded the same payment twice, and read invoices before verifying |
| `call_237a9bc763be` | expired_ssl | `local-realtime` | fail | LFM3-Audio never reached a lookup. Model limit, not in scope here |
| `call_a0fa201a8863` | locked_mailbox | `local-realtime` | invalid | Same |
| `call_cd7a450d7a0f` | expired_ssl | `local-fast` | invalid | LFM3-1.7B as the caller slipped into the agent's role |
| `call_d2a5ddc2e1ec` | locked_mailbox | `local-fast` | invalid | Same |
| `call_360608fd7105` | upload | none | needs_review | Advisory findings on an upload with no scenario. Not a bot result |
| `call_37fee10b707a` | upload | none | needs_review | Same |

How often each cause showed up across the 15 `local-default` calls with a scenario:

| Cause | Calls | Kind |
| --- | --- | --- |
| Changed something before a clear yes, usually chained straight after verifying | 9 | Bot prompt |
| Made up a tool argument: `555-123-4567`, `example.com`, zip `90210`, PIN `1234` | 9 | LLM tool call |
| Read or stated account details before verifying | 5 | Bot prompt |
| A misheard address led to a read-back argument | 5 | STT, then did_you_mean handling |
| Didn't say a fact a rule wants: the date, free, within an hour, the contact email | 4 | Bot prompt |
| Caller left early or skipped a checkpoint | 3 | Caller behavior |
| Caller drifted off script or quoted its sheet | 3 | Caller behavior |
| Tool loop or a copied hold line | 1 | LLM |

One thing ran through nearly every call. The bot spoke, then ran a tool, then spoke again. The gap read as the end of its turn, the caller talked, and the barge-in cost the bot the rest of its turn. Bot turns were cut even on a persona with an interruption rate of 0.

## What changed

Support bot, all in `src/voicelab/agents/`:

- `support.py` gives the model an ordered flow and stays under the prompt's 240 word limit.
- `reminders.py` adds a `next` field to tool results, saying what to do now, in the model's history only. The recorded result and what the judge reads don't change.
- Notes on a round, not kept in history, cover two moments a tool result can't: a caller who failed verification and is pressing for details, and the reply right after verifying.
- Words go out only when a round ends with no tool call. This is the one change with a cost, in the latency numbers below.
- Arguments the caller never said, and zips and PINs with the wrong number of digits, get an error back and never reach the tool. They don't use up a try. This isn't policy: it's bad output from the model, handled like a reply that won't parse.
- A misheard address is swapped for the `did_you_mean` address in later tool calls.
- The prompt and tool descriptions used FH-4419 as an example, which is a real customer's invoice, and the model charged it on someone else's account. They now use an id nobody has.

Tools still enforce no policy. `issue_refund` will refund $189, `pay_invoice` will double charge, and `renew_domain` will charge with no yes. What changed is the descriptions, which are under the 80 character limit, and one new fact in the `issue_refund` result.

Caller bot, in `src/voicelab/agents/caller.py`:

- The script ends a call, not the model. It can leave early only when out of patience.
- Steps advance from what it said where that can be checked: a zip and PIN, a yes, a question, a real opening line. The model used to call a step done at its first half, or forget to, and then replay that step.
- A step that asks has to be asked, even if the bot got there first. A bot that volunteered the new date left a perfect call `invalid`, since the checkpoint says the caller asked.
- Past the last step, only a goodbye ends the call, so "go ahead" stays on the line to hear it done.
- It answers a question first, stays out of the agent's role, and never mentions a sheet or a step.
- Lines stop at a sentence past 24 to 48 words, by persona. Some ran over 30 s.
- In `engine/live/brains.py`, a reply in progress stops if the bot starts speaking again, so the caller doesn't talk over the rest of the bot's turn.

`seeds/policy.md` gained one line saying how long a refund takes, because `scn_double_charge_refund` has the caller ask and nothing could answer it. No scenario's rules, beats, or expected state changed, and neither did the verdict logic in tuning.

## The tuning loop

`scripts/text_loop.py` runs both bots on the real LLM over all 12 scenarios in text, 3 repetitions each, in about 9 minutes. With `--garble` the support bot hears the addresses live STT really misheard. Its scores are proxies, from the state checks and regexes for what the rules want said, so they compare prompts and don't predict verdicts.

| Run | Proxy pass | Required beats reached | Read before verifying | Wrote before verifying |
| --- | --- | --- | --- | --- |
| Baseline | 4 of 36 | 21 of 36 | 30 | 13 |
| After the flow, reminders, held speech, and caller rules | 9 of 36 | 33 of 36 | 4 | 0 |
| Adding the check on made-up arguments | 25 of 36 | 32 of 36 | 0 | 0 |
| Final | 27 of 36 | 31 of 36 | 0 | 0 |

A 12-call live pilot then found what text can't. It was stopped after 5 calls, since the causes were clear, and those 5 stay in `data/`:

- Noisy lines dropped digits. "Five, five, two, one" came through as "552", and "eight, zero, seven, three" as "seventeen", which cost a caller both tries on a PIN it said correctly.
- The bot volunteering an answer left a step unasked.
- An angry persona refused its last step, since its mood outranked the step.
- The bot quoted a charge it had never looked up.

## What the live calls showed

The 36-call run and the held-out comparison are in the [README](../../README.md#held-out-check), and each call's verdicts are in `tests/eval/fixtures/heldout/results.yaml`. Failures that remain, by what the hand labels found:

| Cause | Calls of 36 |
| --- | --- |
| Changed something without a clear yes, or with nobody verified | 3 |
| Refunded past the limit | 1 |
| Read or disclosed account details before verifying | 2 |
| Said something false or never looked up, or said a change was done without doing it | 3 |
| Never offered a way forward, or ended the call on a caller with an open question | 2 |
| Refused a correct caller without running the check | 1 |
| Refunded with no stated amount and no yes | 1 |

## Not tried

- Qwen3.5-9B. There's no GGUF on this machine, so it needs a 5.7 GB download, and the research doc lists it at 34 t/s against 65 t/s. Speech now waits for a whole round, so generation speed adds directly to first audio, and most remaining failures are decisions a prompt or a rule could address. It may still be worth a measured try for the caller alone, where first audio doesn't count.
