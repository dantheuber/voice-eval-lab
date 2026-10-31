"""Load personas and scenarios from `seeds/` and check them against the Fernhill schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from voicelab.company.db import SEEDS_DIR, create_call_db, table_columns
from voicelab.core.models import Persona, Scenario

PERSONAS_DIR = SEEDS_DIR / "personas"
SCENARIOS_DIR = SEEDS_DIR / "scenarios"


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


def scenario_problems(scenario: Scenario) -> list[str]:
    """What Pydantic can't catch: references to customers and columns that don't exist."""
    problems: list[str] = []
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
    finally:
        db.close()
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
