"""Rule-based stand-ins for the three LLM jobs: caller, support agent, and judge.

They read only what a real model gets, the prompt text and message history, so a fake call
exercises the same code paths as a real one. The support rules follow Fernhill's policy closely
enough to reach the expected state of every seeded scenario on a clean line. They are a test
double, not a model of good support.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from voicelab.providers.base import ChatMessage, LLMResponse, ToolCall

MAIL_HOST = "mail.fernhill.example"
REFUND_LIMIT_CENTS = 5000
CONFIRM = "Shall I go ahead?"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Speech splits a domain into words, and only "dot example" marks where it ends.
_DOMAIN = re.compile(r"\bdomain is ((?:[a-z0-9-]+ )*[a-z0-9-]+\.example)\b", re.IGNORECASE)
_PHONE = re.compile(r"\(?\b\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")
_INVOICE = re.compile(r"\bFH-?\s?(\d{4})\b", re.IGNORECASE)
_ZIP_AND_PIN = re.compile(r"\b(\d{5})\b.*?\b(\d{4})\b", re.DOTALL)
_YES = re.compile(r"\b(?:yes|yeah|go ahead|sure|okay|please do)\b", re.IGNORECASE)
_GOODBYE = re.compile(r"\b(?:goodbye|bye)\b", re.IGNORECASE)


# Caller ----------------------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^- (\w+): (.+)$", re.MULTILINE)
_STEP = re.compile(r"Your current step \(id ([^)]+)\): (.+)")
_GOAL = re.compile(r"Why you called: (.+)")
_AGENT_LINE = re.compile(r"^Agent: (.+)$", re.MULTILINE)
_CONDITION = re.compile(r"^(?:When|If) [^,]+, ")
_CLAUSE_BREAK = re.compile(r"\. |, then | then |, and | and (?=ask\b|say\b)")
_ID_FACTS = ("domain", "mailbox", "email_address", "phone")
_NOT_HEARD = "Sorry, what was that?"
_ACCEPT = "Yes, please go ahead."


def _first_person(text: str) -> str:
    text = re.sub(r"\byou're\b", "I'm", text)
    text = re.sub(r"\byou'll\b", "I'll", text)
    text = re.sub(r"\byour\b", "my", text)
    text = re.sub(r"\b(to|telling|tell|at|for|with) you\b", r"\1 me", text)
    return re.sub(r"\byou\b", "I", text)


def _sentence(text: str, end: str = ".") -> str:
    text = text.strip().rstrip(".")
    return f"{text[:1].upper()}{text[1:]}{end}"


@dataclass(frozen=True)
class _Clause:
    kind: str  # "ask", "yes", "say", or "bye"
    text: str = ""


def _parse_clause(clause: str) -> _Clause | None:
    lower = clause.strip()
    lower = lower[:1].lower() + lower[1:]
    if lower.startswith("say yes"):
        return _Clause("yes", _ACCEPT)
    if "goodbye" in lower or "end the call" in lower:
        words = re.sub(r"\s*and end the call$", "", lower)
        if words.startswith("say ") and "goodbye" not in words:
            return _Clause("bye", _sentence(_first_person(words[4:])))
        return _Clause("bye", "Thanks, goodbye.")
    if lower.startswith("say "):
        return _Clause("say", _sentence(_first_person(lower[4:])))
    if lower.startswith("ask the agent to "):
        return _Clause("ask", f"Can you {_first_person(lower[17:])}?")
    if lower.startswith("ask for "):
        return _Clause("ask", f"Can I get {_first_person(lower[8:])}?")
    if lower.startswith("ask "):
        return _Clause("ask", f"Can you tell me {_first_person(lower[4:])}?")
    if lower.startswith("accept"):
        return _Clause("say", "Okay, that's fine.")
    if lower.startswith("grumble"):
        return _Clause("say", "Ugh, fine.")
    if lower.startswith("agree to "):
        return _Clause("yes", _sentence(f"Go ahead and {_first_person(lower[9:])}"))
    if lower.startswith("push"):
        return _Clause("say", "Come on.")
    return None  # "give ..." steps are answered from the fact sheet instead


def _clauses(instruction: str) -> list[_Clause]:
    text = _CONDITION.sub("", instruction.strip().rstrip("."))
    parsed = (_parse_clause(part) for part in _CLAUSE_BREAK.split(text))
    return [clause for clause in parsed if clause]


def _identifier(facts: dict[str, str]) -> str:
    key = next((key for key in _ID_FACTS if key in facts), "")
    if not key:
        return "I don't have that."
    label = "phone number" if key == "phone" else "email" if key != "domain" else "domain"
    return f"My {label} is {facts[key]}."


def _secrets(facts: dict[str, str]) -> str:
    if "billing_zip" in facts and "pin" in facts:
        return f"My billing zip is {facts['billing_zip']} and my PIN is {facts['pin']}."
    return "I don't have that."


def _answer(heard: str, facts: dict[str, str]) -> str | None:
    """A direct answer when the agent asked for something on the sheet."""
    lower = heard.lower()
    if re.search(r"\bzip\b|\bpin\b", lower):
        return _secrets(facts)
    if re.search(r"\bdomain\b|\bphone\b|\bemail\b", lower) and lower.rstrip().endswith("?"):
        return _identifier(facts)
    if "invoice number" in lower:
        return f"It's {facts['invoice_id']}." if "invoice_id" in facts else "I don't have that."
    return None


def _caller_step(beat_id: str, instruction: str, heard: str, facts: dict[str, str]) -> dict:
    answer = _answer(heard, facts)
    if "zip" in instruction.lower():
        if answer is None:
            answer = _identifier(facts) if "?" not in heard else _secrets(facts)
        return _reply(answer, beat_id if answer == _secrets(facts) else None)
    clauses = _clauses(instruction)
    kinds = [clause.kind for clause in clauses]
    offered = CONFIRM.rstrip("?").lower() in heard.lower()
    if answer and "yes" in kinds and not offered:
        return _reply(answer)
    if "yes" in kinds:
        asks_first = "ask" in kinds[: kinds.index("yes")]
        if asks_first and offered:
            clauses = [clause for clause in clauses if clause.kind != "ask"]
        elif asks_first:
            return _reply(" ".join(c.text for c in clauses if c.kind != "yes"))
        elif not offered:
            return _reply(_NOT_HEARD)
    say = " ".join(clause.text for clause in clauses) or answer or "Okay."
    leaves = "bye" in kinds
    return _reply(say, beat_id, hang_up=leaves, reason="finished" if leaves else "")


def _reply(
    say: str, beat: str | None = None, *, hang_up: bool = False, reason: str = ""
) -> dict[str, Any]:
    return {"say": say, "completed_beat": beat, "hang_up": hang_up, "hang_up_reason": reason}


def caller_reply(system: str, request: str) -> dict[str, Any]:
    """Play the caller: one JSON turn built from the fact sheet and the current step."""
    if "You have had enough" in system:
        return _reply("I've had enough of this. Goodbye.", hang_up=True, reason="out of patience")
    sheet = system.split("These are the only facts you know:")[-1].split("If you are asked")[0]
    facts = dict(_FACT_LINE.findall(sheet))
    agent_lines = _AGENT_LINE.findall(request)
    heard = agent_lines[-1] if agent_lines else ""
    step = _STEP.search(system)
    if step is not None:
        return _caller_step(step.group(1), step.group(2), heard, facts)
    if "You got through everything" in system:
        return _reply("Okay, thanks. Bye.", hang_up=True, reason="finished")
    goal = _GOAL.search(system)
    return _reply(_sentence(_first_person(goal.group(1) if goal else "I have a problem")))


# Support agent ---------------------------------------------------------------------------------


@dataclass
class _Call:
    """What the support bot has seen so far, rebuilt from the message history on each request."""

    messages: list[ChatMessage]
    caller_lines: list[str] = field(default_factory=list)
    results: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fresh_results: int = 0  # tool results since the caller last spoke
    last_assistant: str = ""

    def __post_init__(self) -> None:
        for message in self.messages:
            if message.role == "user" and not message.content.startswith("("):
                self.caller_lines.append(message.content)
                self.fresh_results = 0
            elif message.role == "tool":
                self.results.append((message.name or "", _load(message.content)))
                self.fresh_results += 1
            elif message.role == "assistant" and message.content:
                self.last_assistant = message.content

    @property
    def said(self) -> str:
        return self.caller_lines[-1] if self.caller_lines else ""

    @property
    def everything_said(self) -> str:
        return " ".join(self.caller_lines)

    def result(self, name: str) -> dict[str, Any] | None:
        return next((r for tool, r in reversed(self.results) if tool == name), None)

    def count(self, name: str) -> int:
        return sum(tool == name for tool, _ in self.results)

    @property
    def customer_id(self) -> str:
        return (self.result("find_customer") or {}).get("customer_id", "")

    @property
    def consented(self) -> bool:
        return CONFIRM in self.last_assistant and bool(_YES.search(self.said))


def _load(content: str) -> dict[str, Any]:
    try:
        loaded = json.loads(content)
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _tool(name: str, **arguments: Any) -> tuple[str, dict[str, Any]]:
    return name, arguments


@dataclass(frozen=True)
class _Plan:
    """One way of handling a request: look things up, offer it, act on a yes, report back."""

    lookups: tuple[tuple[str, dict[str, Any]], ...] = ()
    offer: str = ""
    actions: tuple[tuple[str, dict[str, Any]], ...] = ()
    report: Callable[[_Call], str] = lambda call: "That's all done."


def _caller_domain(call: _Call) -> str:
    found = _DOMAIN.search(call.everything_said)
    return found.group(1).lower() if found else ""


def _caller_address(call: _Call) -> str:
    found = _EMAIL.search(call.everything_said)
    return found.group().lower() if found else ""


def _first_domain(call: _Call) -> str:
    domains = (call.result("get_account_summary") or {}).get("domains") or [{}]
    return _caller_domain(call) or domains[0].get("name", "")


def _ticket(call: _Call, subject: str) -> tuple[str, dict[str, Any]]:
    summary = f"Caller said: {call.said}"
    return _tool("create_ticket", subject=subject, summary=summary, customer_id=call.customer_id)


def _ticket_report(call: _Call) -> str:
    ticket = (call.result("create_ticket") or {}).get("ticket_id", "")
    return f"Your ticket is {ticket}. A person will answer within one business day."


def _refund_plan(call: _Call) -> _Plan:
    named = _INVOICE.search(call.everything_said)
    listing = call.result("list_invoices")
    if named is None and listing is None:
        return _Plan(lookups=(_tool("list_invoices", customer_id=call.customer_id),))
    paid = [i for i in (listing or {}).get("invoices", []) if i["status"] == "paid"]
    if named:
        invoice_id = f"FH-{named.group(1)}"
    elif paid:
        invoice_id = paid[0]["invoice_id"]
    else:
        return _Plan(
            offer="I can't find a paid invoice to refund. I can open a ticket for you.",
            actions=(_ticket(call, "Refund request"),),
            report=_ticket_report,
        )
    invoice = call.result("get_invoice")
    if invoice is None:
        return _Plan(lookups=(_tool("get_invoice", invoice_id=invoice_id),))
    if invoice.get("status") == "refunded":
        return _Plan(
            offer="That invoice was already refunded. I can open a ticket so a person can check.",
            actions=(_ticket(call, f"Refund dispute on {invoice_id}"),),
            report=_ticket_report,
        )
    if invoice.get("amount_cents", 0) > REFUND_LIMIT_CENTS:
        return _Plan(
            offer="A refund that size needs a person to review it. I can open a ticket for you.",
            actions=(_ticket(call, f"Refund review for {invoice_id}"),),
            report=_ticket_report,
        )
    payments = [p for p in invoice.get("payments", []) if p.get("status") == "succeeded"]
    extra = {"payment_id": payments[-1]["payment_id"]} if len(payments) > 1 else {}
    return _Plan(
        offer=f"I can refund the duplicate charge of {invoice.get('amount')} to your card.",
        actions=(
            _tool(
                "issue_refund",
                invoice_id=invoice_id,
                reason="Duplicate charge" if extra else "Customer request",
                amount_cents=invoice.get("amount_cents"),
                **extra,
            ),
        ),
        report=lambda c: f"Your refund of {invoice.get('amount')} is on its way to your card.",
    )


def _payment_report(call: _Call) -> str:
    payment = call.result("pay_invoice") or {}
    if payment.get("declined"):
        return (
            "Your card on file was declined because it has expired. "
            "You can update it at billing.fernhill.example."
        )
    return f"That payment of {payment.get('amount')} went through."


def _unpaid_plan(call: _Call, *also: tuple[str, dict[str, Any]]) -> _Plan:
    listing = call.result("list_invoices")
    if listing is None:
        lookup = _tool("list_invoices", customer_id=call.customer_id, status="unpaid")
        return _Plan(lookups=(lookup,))
    unpaid = listing.get("invoices", [])
    if not unpaid:
        return _Plan(offer="I don't see anything unpaid. I can open a ticket for you.")
    invoice = unpaid[0]
    return _Plan(
        offer=f"You have {invoice['amount']} unpaid on {invoice['invoice_id']}. "
        "I can charge the card on file.",
        actions=(_tool("pay_invoice", invoice_id=invoice["invoice_id"]), *also),
        report=_payment_report,
    )


def _suspended_plan(call: _Call) -> _Plan:
    return _unpaid_plan(call, _tool("restore_site", domain=_caller_domain(call)))


def _locked_plan(call: _Call) -> _Plan:
    address = _caller_address(call)
    return _Plan(
        offer=f"I can unlock {address}.",
        actions=(_tool("unlock_mailbox", address=address),),
        report=lambda c: "Your mailbox is unlocked. You can sign in again now.",
    )


def _password_plan(call: _Call) -> _Plan:
    address = _caller_address(call)
    return _Plan(
        offer=f"I can send a password reset link for {address} to your contact email.",
        actions=(_tool("reset_mailbox_password", address=address),),
        report=lambda c: "The reset link is on its way to your contact email.",
    )


def _ssl_plan(call: _Call) -> _Plan:
    summary = (_tool("get_account_summary", customer_id=call.customer_id),)
    if call.result("get_account_summary") is None:
        return _Plan(lookups=summary)
    domain = _first_domain(call)
    return _Plan(
        offer=f"The SSL certificate on {domain} has expired. Renewal is free.",
        actions=(_tool("renew_ssl", domain=domain),),
        report=lambda c: "The new certificate will be live in about 15 minutes.",
    )


def _dns_plan(call: _Call) -> _Plan:
    domain = _caller_domain(call)
    records = call.result("get_dns_records")
    if records is None:
        return _Plan(lookups=(_tool("get_dns_records", domain=domain),))
    current = next((r for r in records.get("records", []) if r["type"] == "MX"), {})
    return _Plan(
        offer=f"Your MX record points at {current.get('value')}. I can change it to {MAIL_HOST}.",
        actions=(
            _tool("update_dns_record", domain=domain, type="MX", value=MAIL_HOST, priority=10),
        ),
        report=lambda c: "The change is saved. It usually shows up within an hour.",
    )


def _renewal_plan(call: _Call) -> _Plan:
    domain = _caller_domain(call)
    return _Plan(
        offer=f"Renewing {domain} for one year is $14.99 on the card on file.",
        actions=(_tool("renew_domain", domain=domain, years=1),),
        report=lambda c: (
            f"Your domain now runs until {(c.result('renew_domain') or {}).get('expires_on')}."
        ),
    )


def _bounce_plan(call: _Call) -> _Plan:
    if call.result("get_account_summary") is None:
        return _Plan(lookups=(_tool("get_account_summary", customer_id=call.customer_id),))
    return _Plan(
        offer="Your mailbox is full. Email Plus is $12.00 a month with 25 GB of space.",
        actions=(_tool("change_plan", customer_id=call.customer_id, plan_id="email_plus"),),
        report=lambda c: "You're on Email Plus now, and new mail will arrive again.",
    )


_INTENTS: tuple[tuple[tuple[str, ...], Callable[[_Call], _Plan]], ...] = (
    (("password",), _password_plan),
    (("locked",), _locked_plan),
    (("bouncing", "mailbox is full"), _bounce_plan),
    (("inbound mail", "mx record"), _dns_plan),
    (("charged twice", "two charges", "refund"), _refund_plan),
    (("expiry notice", "renew"), _renewal_plan),
    (("warning", "not private", "certificate"), _ssl_plan),
    (("past due",), _unpaid_plan),
    (("website is down", "site is down"), _suspended_plan),
)


def _say(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop", model="fake-llm")


def _use(call: _Call, name: str, arguments: dict[str, Any], text: str = "") -> LLMResponse:
    tool_call = ToolCall(id=f"call_{len(call.messages)}", name=name, arguments=arguments)
    return LLMResponse(
        text=text, tool_calls=[tool_call], finish_reason="tool_calls", model="fake-llm"
    )


def _identify(call: _Call) -> LLMResponse | None:
    if call.customer_id:
        return None
    text = call.everything_said
    email, domain, phone = _EMAIL.search(text), _DOMAIN.search(text), _PHONE.search(text)
    arguments = {}
    if email:
        arguments = {"email": email.group().lower()}
    elif phone:
        arguments = {"phone": phone.group()}
    elif domain:
        arguments = {"domain": domain.group(1).lower()}
    if arguments and not call.fresh_results:
        return _use(call, "find_customer", arguments, "Let me look that up.")
    return _say("Can I get the domain, phone number, or email on your account?")


def _verify(call: _Call) -> LLMResponse | None:
    if any(r.get("verified") for tool, r in call.results if tool == "verify_identity"):
        return None
    attempts = call.count("verify_identity")
    if attempts >= 2:
        return _say("I can't verify you, so I can't share or change anything. I can open a ticket.")
    details = _ZIP_AND_PIN.search(call.said)
    if details and not call.fresh_results:
        billing_zip, pin = details.groups()
        return _use(
            call, "verify_identity",
            {"customer_id": call.customer_id, "billing_zip": billing_zip, "pin": pin},
        )  # fmt: skip
    if attempts:
        return _say("That doesn't match what I have. What are your billing zip and PIN?")
    return _say("To verify you, what are your billing zip and PIN?")


def support_reply(messages: list[ChatMessage]) -> LLMResponse:
    """Play the agent: identify, verify, look up, offer, act on a yes, and report."""
    call = _Call(messages)
    if _GOODBYE.search(call.said) or call.fresh_results and call.result("end_call"):
        return _use(call, "end_call", {"reason": "caller said goodbye"}, "Thanks for calling.")
    gate = _identify(call) or _verify(call)
    if gate is not None:
        return gate
    spoken = call.everything_said.lower()
    plan_for = next((p for words, p in _INTENTS if any(w in spoken for w in words)), None)
    if plan_for is None:
        return _say("Can you tell me more about the problem?")
    plan = plan_for(call)
    for name, arguments in plan.lookups:
        if not call.result(name):
            return _use(call, name, arguments)
    pending = [(name, args) for name, args in plan.actions if call.result(name) is None]
    if pending:
        if call.consented:
            return _use(call, *pending[0])
        return _say(f"{plan.offer} {CONFIRM}")
    if not plan.actions:
        return _say(plan.offer or "Is there anything else you need today?")
    if call.fresh_results:
        return _say(plan.report(call))
    return _say(f"{plan.report(call)} Is there anything else you need today?")


# Judge -----------------------------------------------------------------------------------------

_TRANSCRIPT_LINE = re.compile(r"^\[[\d.]+s\] (CALLER|BOT): (.+)$", re.MULTILINE)
_QUOTE_WORDS = 10


def judge_reply(prompt: str) -> dict[str, Any]:
    """Pass everything with a transcript behind it, quoting the closest line word for word."""
    lines = _TRANSCRIPT_LINE.findall(prompt)
    if not lines:
        return {"status": "unknown", "reason": "The transcript is empty.", "quotes": []}
    if "The bot must not:" in prompt or "keep to its fact sheet" in prompt:
        reason = "Nothing in the transcript breaks this."
        return {"status": "pass", "reason": reason, "quotes": [], "breaches": []}
    # A script step is about the caller, so only the caller's own words can prove it.
    about_caller = "Step of the script:" in prompt
    candidates = [text for who, text in lines if who == "CALLER" or not about_caller]
    if not candidates:
        return {"status": "unknown", "reason": "The caller said nothing.", "quotes": []}
    wanted = set(re.findall(r"\w+", prompt.rsplit("Transcript:", 1)[-1].lower()))
    best = max(candidates, key=lambda line: len(wanted & set(re.findall(r"\w+", line.lower()))))
    quote = " ".join(best.split()[:_QUOTE_WORDS])
    return {"status": "pass", "reason": "The transcript supports this.", "quotes": [quote]}


# Scenario drafting -----------------------------------------------------------------------------

_CUSTOMER_LINE = re.compile(
    r"^(cus_\d+): ([^,]+), phone ([^,]+), email (\S+), billing_zip (\d+), pin (\d+)\.",
    re.MULTILINE,
)


def draft_reply(system: str, description: str) -> dict[str, Any]:
    """A small valid scenario about the first seeded customer, whatever was asked for."""
    found = _CUSTOMER_LINE.search(system)
    if found is None:
        raise ValueError("draft prompt lists no customers")
    customer_id, name, phone, _, billing_zip, pin = found.groups()
    return {
        "title": description[:60] or "Drafted scenario",
        "description": description,
        "goal": description or "You have a question about your account.",
        "customer_id": customer_id,
        "facts": {"name": name, "phone": phone, "billing_zip": billing_zip, "pin": pin},
        "beats": [
            {
                "id": "state_problem",
                "instruction": "Say why you called.",
                "checkpoint": "The caller explained the problem.",
            },
            {
                "id": "identify",
                "instruction": "Give your phone, then your billing zip and PIN when asked.",
                "checkpoint": "The caller gave identifying details.",
            },
        ],
        "must": [{"id": "verify_first", "text": "Verify the caller.", "severity": "major"}],
        "must_not": [],
        "expected_state": [],
        "pronunciation_watch": [],
        "tags": ["drafted"],
    }
