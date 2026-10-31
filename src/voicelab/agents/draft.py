"""Draft a test scenario from a plain description. Backs the dashboard's "draft with AI" button."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from voicelab.agents.seeds import scenario_problems
from voicelab.agents.speech import unicode_safe
from voicelab.company.db import create_call_db, customer_notes, table_columns
from voicelab.company.policy import load_policy
from voicelab.core.models import Scenario
from voicelab.providers.base import LLM, ChatMessage

_SYSTEM_PROMPT = """\
You write test scenarios for a phone support bot at Fernhill Internet, a company selling email \
hosting, website hosting, domains, and SSL certificates. A caller bot acts out each scenario.

Write the scenario as JSON:
- title: a short name
- description: one sentence on what is being tested
- goal: what the caller wants, in the caller's own words
- customer_id: the seeded customer the caller is, or null for a stranger
- facts: every fact the caller may state, such as name, phone, domain, billing_zip, and pin. \
Copy real values from the customers below. To test failed verification, use wrong values. \
The caller can't invent anything, so include a phone, email, or domain that finds the customer, \
the mailbox, domain, or invoice id at issue, and the correct billing_zip and pin if anything on \
the account should change.
- beats: 3 to 6 ordered steps for the caller, each with an id, instruction, and checkpoint
- must and must_not: what the bot must and must not do, according to the policy. \
Each has an id, text, and severity of critical, major, or minor.
- expected_state: database outcomes using the tables and columns below. Each has an id, table, \
where, op, changes, and description. op is changed, no_change, or row_added.
- pronunciation_watch: tokens the bot must say correctly, such as invoice ids and amounts
- tags: short labels

Tables:
{tables}

Customers:
{customers}

Policy:
{policy}"""


class DraftError(Exception):
    """The LLM never produced a usable scenario."""


def draft_schema() -> dict[str, Any]:
    """The Scenario JSON schema, minus fields the system assigns."""
    schema = Scenario.model_json_schema()
    for assigned in ("id", "created_at", "updated_at"):
        schema["properties"].pop(assigned)
    # Constrained generation drops whatever isn't required. A draft with no customer_id still
    # plays, but with nobody's account behind it, and an empty description shows as a blank field.
    schema["required"] = [
        "title",
        "description",
        "goal",
        "customer_id",
        "facts",
        "beats",
        "must",
        "must_not",
        "expected_state",
    ]
    return schema


def draft_prompt() -> str:
    """The system prompt, with the tables and customers an LLM needs to write valid references."""
    tables = "\n".join(
        f"{table}({', '.join(columns)})" for table, columns in table_columns().items()
    )
    notes = customer_notes()
    db = create_call_db()
    try:
        customers = "\n".join(
            f"{row['id']}: {row['name']}, phone {row['phone']}, email {row['email']}, "
            f"billing_zip {row['billing_zip']}, pin {row['pin']}. {notes.get(row['id'], '')}"
            for row in db.rows("SELECT * FROM customers ORDER BY id")
        )
    finally:
        db.close()
    return _SYSTEM_PROMPT.format(tables=tables, customers=customers, policy=load_policy())


def _check_draft(text: str) -> tuple[Scenario | None, str]:
    """Return the scenario, or None and what to tell the LLM."""
    try:
        data = unicode_safe(json.loads(text))
    except (ValueError, RecursionError) as error:  # also a giant integer or absurd nesting
        return None, f"That was not valid JSON: {error}"
    if not isinstance(data, dict):
        return None, "The reply must be a JSON object."
    fields = {key: value for key, value in data.items() if key in draft_schema()["properties"]}
    try:
        scenario = Scenario.model_validate(fields)
    except ValidationError as error:
        return None, f"That did not match the schema: {error}"
    problems = scenario_problems(scenario)
    if not scenario.beats:
        problems.append("beats must not be empty")
    if not scenario.facts:
        problems.append("facts must not be empty")
    if problems:
        return None, "Fix these problems: " + "; ".join(problems)
    return scenario, ""


async def draft_scenario(
    llm: LLM,
    description: str,
    *,
    attempts: int = 3,
    temperature: float = 0.4,
    max_tokens: int = 2000,
) -> Scenario:
    """Ask the LLM for a Scenario, feeding validation errors back until one passes."""
    messages = [
        ChatMessage(role="system", content=draft_prompt()),
        ChatMessage(role="user", content=f"Write a scenario for this: {description}"),
    ]
    problem = ""
    for _ in range(attempts):
        response = await llm.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_schema=draft_schema()
        )
        scenario, problem = _check_draft(response.text)
        if scenario is not None:
            return scenario
        messages = [
            *messages,
            ChatMessage(role="assistant", content=response.text),
            ChatMessage(role="user", content=problem),
        ]
    raise DraftError(f"no valid scenario after {attempts} attempts: {problem}")
