"""Load personas and scenarios from `seeds/` and check them against the Fernhill schema."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import yaml

from voicelab.company.db import SEEDS_DIR, CompanyDb, create_call_db, table_columns
from voicelab.company.tools import ToolDispatcher
from voicelab.core.models import Persona, Scenario

PERSONAS_DIR = SEEDS_DIR / "personas"
SCENARIOS_DIR = SEEDS_DIR / "scenarios"

# The column a caller would name to point at a row. Other tables hang off the customer.
_CALLER_NAMED_COLUMNS = {
    "mailboxes": "address",
    "websites": "domain",
    "domains": "name",
    "dns_records": "domain",
    "invoices": "id",
    "payments": "invoice_id",
    "refunds": "invoice_id",
}


def load_personas(directory: Path = PERSONAS_DIR) -> dict[str, Persona]:
    return _load(directory, Persona)


def load_scenarios(directory: Path = SCENARIOS_DIR) -> dict[str, Scenario]:
    return _load(directory, Scenario)


def _load[T: (Persona, Scenario)](directory: Path, model: type[T]) -> dict[str, T]:
    loaded: dict[str, T] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        unknown = set(data) - set(model.model_fields)
        if unknown:  # Pydantic would drop a mistyped field and leave the default in place
            raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
        item = model.model_validate(data)
        if item.id in loaded:
            raise ValueError(f"duplicate id {item.id} in {path}")
        loaded[item.id] = item
    return loaded


def _beat_link_problems(scenario: Scenario) -> list[str]:
    beats = {beat.id for beat in scenario.beats}
    rules = [*scenario.must, *scenario.must_not, *scenario.expected_state]
    return [
        f"{rule.id} needs beat {rule.needs_beat}, which isn't in this scenario"
        for rule in rules
        if rule.needs_beat is not None and rule.needs_beat not in beats
    ]


def scenario_problems(scenario: Scenario) -> list[str]:
    """What Pydantic can't catch: references to customers and columns that don't exist."""
    problems = _beat_link_problems(scenario)
    columns = table_columns()
    db = create_call_db()
    try:
        if scenario.customer_id and not db.row(
            "SELECT id FROM customers WHERE id = ?", scenario.customer_id
        ):
            problems.append(f"customer_id {scenario.customer_id} is not a seeded customer")
        for expectation in scenario.expected_state:
            known = columns.get(expectation.table)
            if known is None:
                problems.append(f"expected_state uses unknown table {expectation.table}")
                continue
            for column in [*expectation.where, *expectation.changes]:
                if column not in known:
                    problems.append(f"{expectation.table} has no column {column}")
            if expectation.op == "changed" and all(col in known for col in expectation.where):
                condition = " AND ".join(f"{column} = ?" for column in expectation.where) or "1"
                sql = f"SELECT 1 FROM {expectation.table} WHERE {condition}"
                try:
                    found = db.row(sql, *expectation.where.values())
                except (sqlite3.Error, OverflowError):  # a list or an oversized number
                    problems.append(f"{expectation.where} can't be matched against a row")
                    continue
                if not found:
                    problems.append(
                        f"no seeded {expectation.table} row matches {expectation.where}"
                    )
        if scenario.customer_id:
            problems.extend(_caller_problems(scenario, db))
    finally:
        db.close()
    return problems


def _finds_customer(tools: ToolDispatcher, fact: str, customer_id: str) -> bool:
    """Would a caller reading this fact aloud lead find_customer to the right account?"""
    text = fact.strip()
    if "@" in text:
        key = "email"
    elif re.fullmatch(r"(?:https?://)?[\w-]+(?:\.[\w-]+)+/?", text):
        key = "domain"
    elif len(re.sub(r"\D", "", text)) >= 10:
        key = "phone"
    else:
        return False
    return tools.dispatch("find_customer", {key: text}).get("customer_id") == customer_id


def _caller_problems(scenario: Scenario, db: CompanyDb) -> list[str]:
    """Can a caller who knows only the sheet actually reach the outcome the scenario expects?

    The caller may not invent anything, so a missing fact makes a scenario impossible to pass.
    """
    customer = db.row("SELECT * FROM customers WHERE id = ?", scenario.customer_id)
    if customer is None:
        return []
    facts = list(scenario.facts.values())
    problems = []
    tools = ToolDispatcher(db)
    if not any(_finds_customer(tools, fact, customer["id"]) for fact in facts):
        problems.append(f"facts need a phone, email, or domain that finds {customer['id']}")

    actions = [item for item in scenario.expected_state if item.op != "no_change"]
    if any(item.table != "tickets" for item in actions):  # policy lets anyone open a ticket
        digits = {re.sub(r"\D", "", fact) for fact in facts}
        for label, value in (("billing zip", customer["billing_zip"]), ("PIN", customer["pin"])):
            if value not in digits:
                problems.append(
                    f"expected changes need verification, so facts need the correct {label}"
                )

    sheet = " ".join(facts).lower()
    unpaid = db.rows(
        "SELECT id FROM invoices WHERE customer_id = ? AND status = 'unpaid'", customer["id"]
    )
    for item in actions:
        resource = item.where.get(_CALLER_NAMED_COLUMNS.get(item.table))
        if not isinstance(resource, str) or resource.lower() in sheet:
            continue
        if [invoice["id"] for invoice in unpaid] == [resource]:
            continue  # the bot finds a lone unpaid invoice by itself
        problems.append(f"facts must name {resource} so the caller can point at it")
    return problems


def validate_seeds() -> tuple[dict[str, Persona], dict[str, Scenario]]:
    """Load every seed and raise if a scenario refers to something that doesn't exist."""
    personas = load_personas()
    scenarios = load_scenarios()
    problems = [
        f"{scenario.id}: {problem}"
        for scenario in scenarios.values()
        for problem in scenario_problems(scenario)
    ]
    if problems:
        raise ValueError("invalid seeds:\n" + "\n".join(problems))
    return personas, scenarios
