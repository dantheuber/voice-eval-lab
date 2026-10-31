"""Personas, scenarios with their version history, launcher presets, and key-value settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voicelab.agents.seeds import PERSONAS_DIR, SCENARIOS_DIR, load_personas, load_scenarios
from voicelab.core.models import Persona, Scenario, new_id, utcnow
from voicelab.store.sqlite import Database, timestamp


class Catalog:
    def __init__(self, db: Database) -> None:
        self.db = db

    # Personas ------------------------------------------------------------------------------------

    def personas(self) -> list[Persona]:
        rows = self.db.rows("SELECT data FROM personas ORDER BY name, id")
        return [Persona.model_validate_json(row["data"]) for row in rows]

    def persona(self, persona_id: str) -> Persona | None:
        data = self.db.value("SELECT data FROM personas WHERE id = ?", (persona_id,))
        return Persona.model_validate_json(data) if data else None

    def add_persona(self, persona: Persona) -> Persona:
        self.db.run(
            "INSERT INTO personas (id, name, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                persona.id,
                persona.name,
                persona.model_dump_json(),
                timestamp(persona.created_at),
                timestamp(persona.updated_at),
            ),
        )
        return persona

    def replace_persona(self, persona: Persona) -> Persona | None:
        current = self.persona(persona.id)
        if current is None:
            return None
        saved = persona.model_copy(update={"created_at": current.created_at, "updated_at": utcnow()})
        self.db.run(
            "UPDATE personas SET name = ?, data = ?, updated_at = ? WHERE id = ?",
            (saved.name, saved.model_dump_json(), timestamp(saved.updated_at), saved.id),
        )
        return saved

    def delete_persona(self, persona_id: str) -> bool:
        return self.db.run("DELETE FROM personas WHERE id = ?", (persona_id,)) > 0

    # Scenarios -----------------------------------------------------------------------------------

    def scenarios(self) -> list[tuple[Scenario, int]]:
        rows = self.db.rows("SELECT data, version FROM scenarios ORDER BY title, id")
        return [(Scenario.model_validate_json(row["data"]), row["version"]) for row in rows]

    def scenario(self, scenario_id: str) -> tuple[Scenario, int] | None:
        row = self.db.row("SELECT data, version FROM scenarios WHERE id = ?", (scenario_id,))
        return (Scenario.model_validate_json(row["data"]), row["version"]) if row else None

    def add_scenario(self, scenario: Scenario) -> Scenario:
        data = scenario.model_dump_json()
        with self.db.transaction():
            self.db.run(
                "INSERT INTO scenarios (id, title, version, data, created_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?, ?)",
                (
                    scenario.id,
                    scenario.title,
                    data,
                    timestamp(scenario.created_at),
                    timestamp(scenario.updated_at),
                ),
            )
            self._record_version(scenario.id, 1, data)
        return scenario

    def replace_scenario(self, scenario: Scenario) -> tuple[Scenario, int] | None:
        with self.db.transaction():
            current = self.scenario(scenario.id)
            if current is None:
                return None
            created_at = current[0].created_at
            version = current[1] + 1
            saved = scenario.model_copy(update={"created_at": created_at, "updated_at": utcnow()})
            data = saved.model_dump_json()
            self.db.run(
                "UPDATE scenarios SET title = ?, version = ?, data = ?, updated_at = ? "
                "WHERE id = ?",
                (saved.title, version, data, timestamp(saved.updated_at), saved.id),
            )
            self._record_version(saved.id, version, data)
        return saved, version

    def delete_scenario(self, scenario_id: str) -> bool:
        return self.db.run("DELETE FROM scenarios WHERE id = ?", (scenario_id,)) > 0

    def scenario_versions(self, scenario_id: str) -> list[tuple[int, Scenario]]:
        rows = self.db.rows(
            "SELECT version, data FROM scenario_versions WHERE scenario_id = ? ORDER BY version",
            (scenario_id,),
        )
        return [(row["version"], Scenario.model_validate_json(row["data"])) for row in rows]

    def _record_version(self, scenario_id: str, version: int, data: str) -> None:
        self.db.run(
            "INSERT INTO scenario_versions (scenario_id, version, data, saved_at) "
            "VALUES (?, ?, ?, ?)",
            (scenario_id, version, data, timestamp()),
        )

    # Seeding -------------------------------------------------------------------------------------

    def seed(
        self, personas_dir: Path = PERSONAS_DIR, scenarios_dir: Path = SCENARIOS_DIR
    ) -> dict[str, int]:
        """Insert seeds never seen before. A seed that has been inserted once is never rewritten,
        whether the user edited it, deleted it, or left it alone."""
        added = {"personas": 0, "scenarios": 0}
        for persona in load_personas(personas_dir).values():
            if self._first_seen("persona", persona.id):
                self.add_persona(persona)
                added["personas"] += 1
        for scenario in load_scenarios(scenarios_dir).values():
            if self._first_seen("scenario", scenario.id):
                self.add_scenario(scenario)
                added["scenarios"] += 1
        return added

    def _first_seen(self, kind: str, item_id: str) -> bool:
        return (
            self.db.run(
                "INSERT INTO seeded (kind, id) VALUES (?, ?) ON CONFLICT DO NOTHING",
                (kind, item_id),
            )
            > 0
        )

    # Presets -------------------------------------------------------------------------------------

    def presets(self) -> list[dict[str, Any]]:
        rows = self.db.rows("SELECT id, data FROM presets ORDER BY created_at, id")
        return [{**json.loads(row["data"]), "id": row["id"]} for row in rows]

    def add_preset(self, preset: dict[str, Any]) -> dict[str, Any]:
        preset_id = new_id("preset")
        self.db.run(
            "INSERT INTO presets (id, name, data, created_at) VALUES (?, ?, ?, ?)",
            (preset_id, preset["name"], json.dumps(preset), timestamp()),
        )
        return {**preset, "id": preset_id}

    # Settings ------------------------------------------------------------------------------------

    def setting(self, key: str, default: Any = None) -> Any:
        raw = self.db.value("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(raw) if raw is not None else default

    def set_setting(self, key: str, value: Any) -> None:
        self.db.run(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
