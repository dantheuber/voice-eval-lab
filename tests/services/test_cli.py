from __future__ import annotations

import shutil

import pytest

from voicelab import cli
from voicelab.services.settings import REPO_ROOT


def test_serve_options_parse():
    arguments = cli._parser().parse_args(["serve", "--http-port", "1", "--services", "lazy"])
    assert (arguments.command, arguments.http_port, arguments.services) == ("serve", 1, "lazy")


def test_run_takes_scenario_and_persona():
    arguments = cli._parser().parse_args(["run", "scn_a", "per_a", "--mode", "turn_based"])
    assert (arguments.scenario, arguments.persona, arguments.mode) == (
        "scn_a",
        "per_a",
        "turn_based",
    )


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli._parser().parse_args([])


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Only the fake profile, because the real ones would start real model servers."""
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    shutil.copy(REPO_ROOT / "config" / "profiles" / "fake.yaml", profiles)
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOICELAB_CERTS_DIR", str(tmp_path / "certs"))
    monkeypatch.setenv("VOICELAB_PROFILES_DIR", str(profiles))
    return tmp_path


def test_seed_loads_and_is_idempotent(isolated, capsys):
    assert cli.main(["seed"]) == 0
    assert "'personas': 9" in capsys.readouterr().out
    assert cli.main(["seed"]) == 0
    assert "'personas': 0" in capsys.readouterr().out


def test_eval_of_a_missing_call_fails_cleanly(isolated, capsys):
    assert cli.main(["eval", "call_missing"]) == 1
    assert "no call" in capsys.readouterr().err


def test_run_and_eval_on_the_fake_profile(isolated, capsys):
    code = cli.main(
        ["run", "scn_dns_mx_fix", "per_sysadmin", "--profile", "fake", "--mode", "turn_based"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "started call_" in out
    call_id = next(line.split()[1] for line in out.splitlines() if line.startswith("started"))
    assert f"{call_id}: done" in out
    assert cli.main(["eval", call_id]) == 0
    assert "Verdict:" in capsys.readouterr().out


def test_one_shot_commands_leave_the_active_profiles_servers_alone(isolated, capsys):
    """A real profile being active must not mean `run --profile fake` loads every model."""
    busy = isolated / "profiles" / "busy.yaml"
    marker = isolated / "server-started"
    busy.write_text(
        "id: local-default\nlabel: Busy\n"
        "roles: {}\n"
        "services:\n"
        "  - name: heavy\n"
        "    port: 18996\n"
        f"    command: [touch, {marker}]\n"
    )
    code = cli.main(
        ["run", "scn_dns_mx_fix", "per_sysadmin", "--profile", "fake", "--mode", "turn_based"]
    )
    assert code == 0, capsys.readouterr().out
    assert not marker.exists()
