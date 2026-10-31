#!/usr/bin/env python3
"""The fast tuning loop: both bots on the real LLM over all seed scenarios, in text, with no audio.

Scores are proxies, not verdicts. State comes from the real state evaluator's checks, and rules
come from the regexes below, so use this to compare prompt changes and confirm on live calls.
With `--garble`, the support bot hears the addresses the way live STT really misheard them.

    uv run python scripts/text_loop.py --label try1 --reps 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from voicelab.agents.caller import CallerBot, TextCallResult, run_text_call
from voicelab.agents.seeds import load_personas, load_scenarios
from voicelab.agents.support import SupportBot
from voicelab.company.db import create_call_db
from voicelab.company.tools import ToolDispatcher
from voicelab.core.models import Scenario, Speaker, SystemEvent, ToolCallEvent, Utterance
from voicelab.eval.evaluators.state import satisfying_changes
from voicelab.providers.adapters.openai_compat import OpenAICompatLLM

READS = ("get_invoice", "list_invoices", "get_account_summary", "get_dns_records")
WRITES = (
    "issue_refund",
    "pay_invoice",
    "restore_site",
    "renew_ssl",
    "unlock_mailbox",
    "reset_mailbox_password",
    "update_dns_record",
    "renew_domain",
    "change_plan",
)
YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|ok|go ahead|please do|do it|sounds good|fine|alright|charge it"
    r"|correct|right)\b",
    re.IGNORECASE,
)
# Live STT mishearings from real calls.
GARBLE = {
    "hartfamily": "heartfamily",
    "nairdesign": "neardesign",
    "nguyenauto": "winauto",
    "marinocafe": "merinocafe",
    "kowalskicrafts": "koelskicrafts",
    "thompsonphoto": "thomsonphoto",
}
# What each scenario's `must` rules need the bot to say, near enough for a regex.
SAY: dict[str, dict[str, str]] = {
    "scn_already_refunded": {"tells_truth": r"(already|was) refunded", "ticket": r"ticket"},
    "scn_dns_mx_fix": {"timing": r"(an|one) hour[\s\S]*(24|twenty.?four)"},
    "scn_domain_renewal": {"new_date": r"november (6|sixth)"},
    "scn_double_charge_refund": {},
    "scn_expired_ssl": {"cause": r"october (21|twenty.?(first|one))", "free": r"free|no charge"},
    "scn_failed_verification": {"next": r"ticket"},
    "scn_forgotten_password": {"destination": r"priya[. ]*(dot )?nair (at|@)|mailbox dot example"},
    "scn_locked_mailbox": {},
    "scn_past_due_expired_card": {"declined": r"expired|declined", "site": r"billing dot fernhill"},
    "scn_plan_upgrade_full_mailbox": {
        "cause": r"\bfull\b",
        "quote": r"(12|twelve)[\s\S]*(25|twenty.?five)|(25|twenty.?five)[\s\S]*(12|twelve)",
    },
    "scn_refund_over_limit": {"expectation": r"business day"},
    "scn_site_suspended_nonpayment": {"cause": r"suspend", "amount": r"29\.99|twenty.?nine"},
}
# A change that needs a yes on the caller's last turn, and what must be said first.
QUOTE_BEFORE = {
    "renew_domain": r"14\.99|fourteen",
    "issue_refund": r"49\.99|forty.?nine",
    "pay_invoice": r"\$|dollars",
    "change_plan": r"12|twelve",
}


class HearsLikeStt(SupportBot):
    """Hands the bot the words live STT made of the caller's line."""

    def respond(self, caller_text: str):
        for right, wrong in GARBLE.items():
            caller_text = re.sub(right, wrong, caller_text, flags=re.IGNORECASE)
        return super().respond(caller_text)


@dataclass
class Score:
    scenario: str
    persona: str
    rep: int
    end_reason: str
    detail: str
    turns: int
    beats_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _spoken_invention(event: ToolCallEvent, heard: str) -> list[str]:
    """Arguments the caller never said, as far as digits and domain words can show."""
    invented = []
    wanted = {
        "find_customer": ("phone", "email", "domain"),
        "verify_identity": ("billing_zip", "pin"),
    }.get(event.name, ())
    plain = re.sub(r"[^a-z0-9]", "", heard.lower())
    for name in wanted:
        value = re.sub(r"[^a-z0-9]", "", str(event.arguments.get(name) or "").lower())
        if value and value[:4] not in plain:
            invented.append(f"{name}={event.arguments[name]}")
    return invented


def score(
    scenario: Scenario, persona_id: str, rep: int, result: TextCallResult, garble: bool
) -> Score:
    required = [beat.id for beat in scenario.beats if beat.required]
    done = result.completed_beats
    outcome = Score(
        scenario.id,
        persona_id,
        rep,
        result.end_reason,
        result.detail,
        sum(isinstance(e, Utterance) and e.speaker == Speaker.CALLER for e in result.timeline),
        all(beat in done for beat in required),
    )
    problems = outcome.problems
    if not outcome.beats_ok:
        missed = [beat for beat in required if beat not in done]
        problems.append(f"beats:{','.join(missed)}")

    verified = False
    bot_so_far = caller_so_far = from_tools = last_caller = ""
    for event in result.timeline:
        if isinstance(event, Utterance):
            text = event.spoken_text or ""
            if event.speaker == Speaker.CALLER:
                caller_so_far += f" {text}"
                last_caller = text
            else:
                bot_so_far += f" {text}"
        elif isinstance(event, ToolCallEvent):
            heard = f"{caller_so_far} {from_tools}"
            if garble:
                for right, wrong in GARBLE.items():
                    heard = re.sub(right, wrong, heard, flags=re.IGNORECASE)
            problems += [f"invented:{item}" for item in _spoken_invention(event, heard)]
            from_tools += f" {json.dumps(event.result)}"
            if event.name == "verify_identity" and event.result.get("verified") is True:
                verified = True
            elif not verified and not event.error and event.name in READS:
                problems.append(f"read_unverified:{event.name}")
            elif not event.error and event.name in WRITES:
                if not verified:
                    problems.append(f"write_unverified:{event.name}")
                if not YES.search(last_caller):
                    problems.append(f"write_no_yes:{event.name}")
                quote = QUOTE_BEFORE.get(event.name)
                if quote and not re.search(quote, bot_so_far, re.IGNORECASE):
                    problems.append(f"write_no_quote:{event.name}")

    for expectation in scenario.expected_state:
        changes = satisfying_changes(expectation, result.db_changes)
        met = not changes if expectation.op == "no_change" else bool(changes)
        if not met:
            problems.append(f"state:{expectation.id}")
    for rule, pattern in SAY[scenario.id].items():
        if not re.search(pattern, bot_so_far, re.IGNORECASE):
            problems.append(f"said:{rule}")
    return outcome


def transcript(result: TextCallResult) -> str:
    lines = []
    for event in result.timeline:
        if isinstance(event, Utterance):
            who = "C" if event.speaker == Speaker.CALLER else "B"
            lines.append(f"{who}: {event.spoken_text}{' [CUT]' if event.interrupted else ''}")
        elif isinstance(event, ToolCallEvent):
            lines.append(
                f"T: {event.name} {json.dumps(event.arguments)} -> {json.dumps(event.result)[:200]}"
            )
        elif isinstance(event, SystemEvent) and event.kind not in ("call_started", "call_ended"):
            lines.append(f"S: {event.kind} {event.detail[:120]}")
    return "\n".join(lines)


async def one_call(args, llm, scenario, persona, rep, out: Path) -> Score:
    db = create_call_db()
    try:
        bot_type = HearsLikeStt if args.garble else SupportBot
        support = bot_type(llm, ToolDispatcher(db))
        caller = CallerBot(llm, persona, scenario, seed=rep)
        result = await run_text_call(support, caller)
        outcome = score(scenario, persona.id, rep, result, args.garble)
    finally:
        db.close()
    head = (
        f"{outcome.scenario} {outcome.persona} rep {rep}: {outcome.end_reason} ({outcome.detail})"
    )
    body = transcript(result)
    (out / f"{scenario.id}.{persona.id}.{rep}.txt").write_text(
        f"{head}\nPROBLEMS: {outcome.problems}\n\n{body}\n"
    )
    print(f"{'ok ' if outcome.ok else 'BAD'} {head} {outcome.problems}", flush=True)
    return outcome


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--garble", action="store_true")
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--base-url", default="http://127.0.0.1:18102/v1")
    parser.add_argument("--out", default="/tmp/vl/runs")
    args = parser.parse_args()

    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    llm = OpenAICompatLLM(args.base_url, name="text-loop")
    scenarios = list(load_scenarios().values())
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only.split(",")]
    personas = list(load_personas().values())

    gate = asyncio.Semaphore(args.jobs)
    outcomes: list[Score] = []

    async def guarded(scenario, persona, rep):
        async with gate:
            outcomes.append(await one_call(args, llm, scenario, persona, rep, out))

    await asyncio.gather(
        *(
            guarded(scenario, personas[(index * 3 + rep) % len(personas)], rep)
            for index, scenario in enumerate(scenarios)
            for rep in range(args.reps)
        )
    )

    by_scenario: dict[str, list[Score]] = defaultdict(list)
    for outcome in outcomes:
        by_scenario[outcome.scenario].append(outcome)
    for scenario_id, items in sorted(by_scenario.items()):
        passed = sum(item.ok for item in items)
        print(f"{scenario_id:36} {passed}/{len(items)}")
    kinds = Counter(p.split(":")[0] for o in outcomes for p in o.problems)
    print("problems:", dict(kinds.most_common()))
    print("end:", dict(Counter(o.end_reason for o in outcomes)))
    print("caller turns, median:", statistics.median(o.turns for o in outcomes))
    print(
        f"TOTAL {sum(o.ok for o in outcomes)}/{len(outcomes)}, "
        f"beats ok {sum(o.beats_ok for o in outcomes)}/{len(outcomes)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
