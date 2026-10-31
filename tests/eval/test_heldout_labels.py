"""The held-out labels are a record of a measurement, so they have to stay complete and readable."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from voicelab.agents.seeds import load_scenarios

LABELS = yaml.safe_load((Path(__file__).parent / "fixtures/heldout/labels.yaml").read_text())
CALLS = LABELS["calls"]
VERDICTS = {"pass", "fail", "needs_review", "invalid"}


def test_every_scenario_has_three_calls():
    counts = Counter(call["scenario"] for call in CALLS.values())
    assert counts == dict.fromkeys(load_scenarios(), 3)


def test_every_label_is_a_verdict_with_a_reason():
    for call_id, call in CALLS.items():
        assert call["verdict"] in VERDICTS, call_id
        assert call["why"].strip(), call_id


def test_the_labels_include_every_kind_of_call_the_eval_can_get_wrong():
    assert {"pass", "fail", "invalid"} <= {call["verdict"] for call in CALLS.values()}
