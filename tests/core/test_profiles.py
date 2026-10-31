"""Profiles: the settings a stack carries beside its adapters."""

import pytest
import yaml
from pydantic import ValidationError

from voicelab.core.profiles import Profile, default_profiles_dir, load_profiles


def _profile(**extra) -> Profile:
    return Profile.model_validate({"id": "x", "label": "X", "roles": {}, **extra})


def test_the_argument_guard_is_on_unless_a_profile_says_otherwise():
    assert _profile().support_bot.argument_guard is True


def test_a_profile_can_turn_the_argument_guard_off():
    text = "id: x\nlabel: X\nroles: {}\nsupport_bot:\n  argument_guard: false\n"
    assert Profile.model_validate(yaml.safe_load(text)).support_bot.argument_guard is False


def test_a_misspelled_switch_is_refused_rather_than_testing_the_wrong_bot():
    with pytest.raises(ValidationError, match="argument_gaurd"):
        _profile(support_bot={"argument_gaurd": False})


def test_every_shipped_profile_keeps_the_guard_on():
    profiles = load_profiles(default_profiles_dir())
    assert profiles
    assert all(profile.support_bot.argument_guard for profile in profiles.values())


def test_the_profiles_people_tune_show_the_switch():
    for name in ("local-default", "local-fast"):
        data = yaml.safe_load((default_profiles_dir() / f"{name}.yaml").read_text())
        assert data["support_bot"] == {"argument_guard": True}
