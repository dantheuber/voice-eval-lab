from __future__ import annotations

import pytest

from voicelab.core.models import Persona, Scenario, VoiceRef
from voicelab.store.calls import BadCursor, CallFilters, CallIndex
from voicelab.store.catalog import Catalog
from voicelab.store.runs import RunStore
from voicelab.store.sqlite import Database, timestamp


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.sqlite3")
    yield database
    database.close()


@pytest.fixture
def catalog(db):
    return Catalog(db)


def write_seeds(tmp_path, name="Seeded"):
    personas, scenarios = tmp_path / "p", tmp_path / "s"
    personas.mkdir(exist_ok=True)
    scenarios.mkdir(exist_ok=True)
    (personas / "a.yaml").write_text(f"id: per_a\nname: {name}\nvoice:\n  voice_id: v\n")
    (scenarios / "a.yaml").write_text(f"id: scn_a\ntitle: {name}\ngoal: g\n")
    return personas, scenarios


def test_real_seeds_load(catalog):
    added = catalog.seed()
    assert added["personas"] == 9
    assert added["scenarios"] == 12


def test_seeding_twice_adds_nothing(catalog):
    catalog.seed()
    assert catalog.seed() == {"personas": 0, "scenarios": 0}
    assert len(catalog.personas()) == 9


def test_seeding_keeps_user_edits(catalog, tmp_path):
    dirs = write_seeds(tmp_path)
    catalog.seed(*dirs)
    persona = catalog.persona("per_a")
    catalog.replace_persona(persona.model_copy(update={"name": "Edited"}))
    scenario, _ = catalog.scenario("scn_a")
    catalog.replace_scenario(scenario.model_copy(update={"title": "Edited"}))
    catalog.seed(*write_seeds(tmp_path, name="Changed in the repo"))
    assert catalog.persona("per_a").name == "Edited"
    assert catalog.scenario("scn_a")[0].title == "Edited"
    assert catalog.scenario("scn_a")[1] == 2


def test_seeding_does_not_bring_back_what_the_user_deleted(catalog, tmp_path):
    dirs = write_seeds(tmp_path)
    catalog.seed(*dirs)
    catalog.delete_persona("per_a")
    catalog.delete_scenario("scn_a")
    assert catalog.seed(*dirs) == {"personas": 0, "scenarios": 0}
    assert catalog.persona("per_a") is None


def test_a_seed_added_later_is_picked_up(catalog, tmp_path):
    personas, scenarios = write_seeds(tmp_path)
    catalog.seed(personas, scenarios)
    (personas / "b.yaml").write_text("id: per_b\nname: New\nvoice:\n  voice_id: v\n")
    assert catalog.seed(personas, scenarios) == {"personas": 1, "scenarios": 0}


def test_edits_survive_reopening_the_database(tmp_path):
    path = tmp_path / "x.sqlite3"
    first = Database(path)
    catalog = Catalog(first)
    persona = catalog.add_persona(Persona(name="Ann", voice=VoiceRef(voice_id="v")))
    catalog.set_setting("thing", {"a": 1})
    first.close()
    second = Database(path)
    reopened = Catalog(second)
    assert reopened.persona(persona.id).name == "Ann"
    assert reopened.setting("thing") == {"a": 1}
    second.close()


def test_every_scenario_version_is_kept(catalog):
    scenario = catalog.add_scenario(Scenario(title="One", goal="g"))
    catalog.replace_scenario(scenario.model_copy(update={"title": "Two"}))
    catalog.replace_scenario(scenario.model_copy(update={"title": "Three"}))
    titles = [s.title for _, s in catalog.scenario_versions(scenario.id)]
    assert titles == ["One", "Two", "Three"]
    assert catalog.scenario(scenario.id)[1] == 3


def test_replacing_something_missing_returns_none(catalog):
    assert catalog.replace_persona(Persona(name="n", voice=VoiceRef(voice_id="v"))) is None
    assert catalog.replace_scenario(Scenario(title="t", goal="g")) is None


def test_settings_default(catalog):
    assert catalog.setting("missing", 5) == 5


def _call(index, call_id, **columns):
    index.add(id=call_id, mode="turn_based", status="done", timing_is_real=0, **columns)


def test_transaction_rolls_back(db):
    index = CallIndex(db)
    with pytest.raises(RuntimeError), db.transaction():
        _call(index, "call_1")
        raise RuntimeError
    assert index.get("call_1") is None


def test_sql_values_are_bound_not_spliced(db):
    index = CallIndex(db)
    _call(index, "call_1", verdict="pass")
    filters = CallFilters(scenario=["x' OR '1'='1"])
    assert index.page(filters, None, 10).total == 0


def test_bad_cursors_raise_bad_cursor(db):
    with pytest.raises(BadCursor):
        CallIndex(db).page(CallFilters(), "!!!", 10)
    with pytest.raises(BadCursor):
        RunStore(db).page("bm90IGpzb24=", 10)


def test_calls_per_hour_bins_by_age(db):
    from datetime import UTC, datetime, timedelta

    index = CallIndex(db)
    now = datetime(2026, 10, 30, 12, 30, tzinfo=UTC)
    _call(index, "call_now", created_at=timestamp(now - timedelta(minutes=5)))
    _call(index, "call_earlier", created_at=timestamp(now - timedelta(hours=3, minutes=5)))
    _call(index, "call_old", created_at=timestamp(now - timedelta(hours=30)))
    bins = index.calls_per_hour(None, now)
    assert len(bins) == 24
    assert bins[23] == 1
    assert bins[20] == 1
    assert sum(bins) == 2


def test_run_uses_only_counts_unfinished_runs(db):
    runs = RunStore(db)
    pairs = [{"scenario_id": "scn_a", "persona_id": "per_a"}]
    fields = dict(
        label="l",
        profile="p",
        mode="turn_based",
        started_at=timestamp(),
        reps=1,
        concurrency=1,
        pairs=pairs,
    )
    runs.add(id="batch_done", status="done", **fields)
    assert not runs.uses("persona_id", "per_a")
    runs.add(id="batch_paused", status="paused", **fields)
    assert runs.uses("persona_id", "per_a")
    assert runs.uses("scenario_id", "scn_a")
