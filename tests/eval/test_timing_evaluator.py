import pytest
from evalfakes import write_call

from voicelab.core.models import (
    BargeInEvent,
    CallArtifact,
    CallMode,
    Speaker,
    SystemEvent,
    Utterance,
)
from voicelab.eval.config import EvalConfig
from voicelab.eval.context import EvalContext, EvalDeps, EvaluatorOutput
from voicelab.eval.evaluators.timing import (
    TimingEvaluator,
    find_barge_ins,
    find_overlaps,
    find_silences,
    longest_monologue,
    merge_turns,
    response_latencies,
    unanswered_turns,
)
from voicelab.eval.vad import Region
from voicelab.store.artifacts import SEPARATION_EVENT

TOLERANCE = 0.03


async def evaluate(
    store, artifact=None, config=None, **spans
) -> tuple[EvaluatorOutput, CallArtifact]:
    artifact, paths = write_call(store, artifact, **spans)
    ctx = EvalContext(artifact, paths, EvalDeps(config=config or EvalConfig()))
    out = EvaluatorOutput()
    await TimingEvaluator().run(ctx, out)
    return out, artifact


def categories(out: EvaluatorOutput) -> list[str]:
    return [f.category for f in out.findings]


def regions(*pairs: tuple[float, float]) -> list[Region]:
    return [Region(s, e) for s, e in pairs]


def test_overlap_records_who_started_it():
    overlaps = find_overlaps(regions((0, 4)), regions((3, 5), (5.5, 6)))
    assert len(overlaps) == 1
    assert (overlaps[0].start_s, overlaps[0].end_s) == (3, 4)
    assert overlaps[0].started_by == Speaker.BOT
    assert find_overlaps(regions((3, 5)), regions((0, 4)))[0].started_by == Speaker.CALLER


def test_overlap_ignores_touching_turns_and_short_blips():
    assert find_overlaps(regions((0, 2)), regions((2, 3))) == []
    assert find_overlaps(regions((0, 2)), regions((1.95, 3)), min_s=0.1) == []


def test_silences_exclude_time_before_the_first_speech():
    silences = find_silences(regions((2, 3), (8, 9)), regions((4, 5)), end_s=9)
    assert silences == regions((3, 4), (5, 8))


def test_silences_handle_nested_speech():
    assert find_silences(regions((0, 10)), regions((2, 3)), end_s=10) == []


def test_merge_turns_joins_short_gaps_only():
    assert merge_turns(regions((0, 1), (1.5, 2), (3, 4)), 0.7) == regions((0, 2), (3, 4))


def test_monologue_stops_at_an_interjection():
    longest = longest_monologue(regions((0, 10)), regions((6, 7)), 1.5)
    assert longest == Region(0, 6)


def test_monologue_bridges_own_pauses():
    assert longest_monologue(regions((0, 3), (4, 6)), [], 1.5) == Region(0, 6)


def test_response_latency_pairs_each_answered_turn():
    latencies = response_latencies(regions((0, 2), (5, 6)), regions((2.5, 4), (7, 8)))
    assert [round(x.latency_s, 3) for x in latencies] == [0.5, 1.0]


def test_unanswered_turn_has_no_latency():
    latencies = response_latencies(regions((0, 1), (3, 4)), regions((5, 6)))
    assert [(x.caller_end_s, x.bot_start_s) for x in latencies] == [(4, 5)]


def test_barge_in_measures_caller_start_to_bot_stop():
    barge_ins = find_barge_ins(regions((3, 5)), regions((0, 3.4)), min_overlap_s=0.1)
    assert len(barge_ins) == 1
    assert barge_ins[0].reaction_s == pytest.approx(0.4)


async def test_metrics_on_constructed_overlapping_audio(store):
    # caller 0-2, bot 2.5-6, caller cuts in at 5.0 and bot stops at 5.4, caller to 7, bot 8-9
    out, _ = await evaluate(
        store,
        caller=[(0.0, 2.0), (5.0, 7.0)],
        bot=[(2.5, 5.4), (8.0, 9.0)],
        duration_s=10.0,
    )
    m = out.metrics
    assert m["caller_talk_s"] == pytest.approx(4.0, abs=2 * TOLERANCE)
    assert m["bot_talk_s"] == pytest.approx(3.9, abs=2 * TOLERANCE)
    assert m["talk_ratio_bot"] == pytest.approx(3.9 / 7.9, abs=0.01)

    assert m["overlap"]["count"] == 1
    assert m["overlap"]["total_s"] == pytest.approx(0.4, abs=TOLERANCE)
    assert m["overlap"]["caller_started"] == 1
    assert m["overlap"]["bot_started"] == 0

    latency = m["response_latency"]
    assert [t["latency_s"] for t in latency["turns"]] == pytest.approx([0.5, 1.0], abs=TOLERANCE)
    assert latency["max_s"] == pytest.approx(1.0, abs=TOLERANCE)

    events = m["barge_in"]["events"]
    assert len(events) == 1
    assert events[0]["caller_start_s"] == pytest.approx(5.0, abs=TOLERANCE)
    assert events[0]["reaction_s"] == pytest.approx(0.4, abs=TOLERANCE)

    assert m["dead_air"]["longest_s"] == pytest.approx(1.0, abs=TOLERANCE)
    assert m["longest_monologue"]["bot"]["duration_s"] == pytest.approx(2.5, abs=TOLERANCE)
    assert out.findings == []


async def test_dead_air_finding_and_severity(store):
    out, _ = await evaluate(store, caller=[(0, 1)], bot=[(8, 9), (31, 32)], duration_s=32.0)
    dead = [f for f in out.findings if f.category == "dead_air"]
    assert [f.severity for f in dead] == ["minor", "major"]
    assert dead[1].evidence[0].start_s == pytest.approx(9.0, abs=TOLERANCE)
    assert out.metrics["dead_air"]["count_over_threshold"] == 2


async def test_slow_response_finding_uses_configured_threshold(store):
    spans = {"caller": [(0, 1)], "bot": [(3, 4)], "duration_s": 5.0}
    out, _ = await evaluate(store, **spans)
    assert "slow_response" not in categories(out)
    config = EvalConfig()
    config.timing.slow_response_s = 1.5
    out, _ = await evaluate(store, config=config, **spans)
    assert "slow_response" in categories(out)


async def test_very_slow_response_is_major(store):
    out, _ = await evaluate(store, caller=[(0, 1)], bot=[(10, 11)], duration_s=12.0)
    slow = next(f for f in out.findings if f.category == "slow_response")
    assert slow.severity == "major"
    assert slow.title == "Bot took 9.0 s to respond"


async def test_bot_talking_over_the_caller_is_found(store):
    out, _ = await evaluate(store, caller=[(0, 5)], bot=[(1, 4)], duration_s=6.0)
    over = next(f for f in out.findings if f.category == "talk_over")
    assert over.severity == "major"
    assert over.evidence[0].start_s == pytest.approx(1.0, abs=TOLERANCE)


async def test_slow_reaction_to_barge_in_is_found(store):
    out, _ = await evaluate(store, caller=[(2, 6)], bot=[(0, 4)], duration_s=7.0)
    slow = next(f for f in out.findings if f.category == "slow_barge_in")
    assert "2.0 s" in slow.title


async def test_quick_reaction_to_barge_in_is_not_a_finding(store):
    out, _ = await evaluate(store, caller=[(2, 6)], bot=[(0, 2.3)], duration_s=7.0)
    assert out.findings == []
    assert out.metrics["barge_in"]["events"][0]["reaction_s"] == pytest.approx(0.3, abs=TOLERANCE)


async def test_synthetic_timing_skips_latency_and_barge_in_and_says_why(store):
    artifact = CallArtifact(mode=CallMode.TURN_BASED, timing_is_real=False)
    out, _ = await evaluate(
        store, artifact, caller=[(0, 1), (9, 11)], bot=[(12, 13), (30, 35)], duration_s=36.0
    )
    m = out.metrics
    assert m["response_latency"] is None
    assert m["barge_in"] is None
    assert "turn-based" in m["skipped"]["latency"]
    assert "turn-based" in m["skipped"]["barge_in"]
    assert m["dead_air"]["synthetic"] is True
    assert out.findings == []
    assert m["talk_ratio_bot"] is not None


async def test_inferred_channels_skip_overlap_and_barge_in(store):
    artifact = CallArtifact(
        mode=CallMode.UPLOAD,
        timeline=[SystemEvent(at_s=0, kind=SEPARATION_EVENT, detail="split")],
    )
    out, _ = await evaluate(store, artifact, caller=[(0, 2)], bot=[(3, 4)], duration_s=5.0)
    assert out.metrics["overlap"] is None
    assert out.metrics["barge_in"] is None
    assert "mono" in out.metrics["skipped"]["overlap"]
    assert out.metrics["response_latency"]["max_s"] == pytest.approx(1.0, abs=TOLERANCE)


async def test_engine_reports_sit_beside_audio_measurements(store):
    artifact = CallArtifact(
        mode=CallMode.HUMAN,
        timeline=[
            Utterance(speaker=Speaker.BOT, start_s=2.5, end_s=5.4, ttfb_ms=480.0),
            BargeInEvent(at_s=5.0, bot_stopped_at_s=5.4),
        ],
    )
    out, _ = await evaluate(
        store, artifact, caller=[(0, 2), (5, 7)], bot=[(2.5, 5.4)], duration_s=8.0
    )
    assert out.metrics["response_latency"]["reported_ttfb_median_ms"] == 480.0
    assert out.metrics["barge_in"]["reported_by_engine"] == 1


async def test_silent_call_produces_empty_metrics(store):
    out, _ = await evaluate(store, duration_s=3.0)
    assert out.metrics["talk_ratio_bot"] is None
    assert out.metrics["longest_monologue"] == {"caller": None, "bot": None}
    assert out.findings == []


# Turns end where the other side speaks, and silence runs to the end of the call ------------------


def test_turns_do_not_merge_across_the_other_speaker():
    assert merge_turns(regions((0, 1), (1.6, 5)), 0.7, regions((1.2, 6))) == regions(
        (0, 1), (1.6, 5)
    )


def test_turns_merge_when_the_other_speaker_is_outside_the_gap():
    assert merge_turns(regions((0, 1), (1.6, 5)), 0.7, regions((6, 7))) == regions((0, 5))


async def test_caller_speaking_in_a_bot_pause_is_not_a_barge_in(store):
    out, _ = await evaluate(store, bot=[(0, 1), (1.6, 5)], caller=[(1.2, 6)], duration_s=7.0)
    assert out.metrics["barge_in"]["events"] == []
    assert "slow_barge_in" not in categories(out)
    # The bot really did come back on top of the caller, and that part is true.
    over = next(f for f in out.findings if f.category == "talk_over")
    assert over.title == "Bot talked over the caller for 3.4 s"


async def test_a_bot_reply_in_a_caller_pause_keeps_its_latency(store):
    out, _ = await evaluate(store, caller=[(0, 1), (1.6, 2)], bot=[(1.1, 1.5)], duration_s=4.0)
    turns = out.metrics["response_latency"]["turns"]
    assert [t["latency_s"] for t in turns] == pytest.approx([0.1], abs=TOLERANCE)


def test_barge_in_is_measured_against_the_bot_region_the_caller_cut_into():
    barge_ins = find_barge_ins(regions((2, 3)), regions((0, 2.4), (2.8, 5)), min_overlap_s=0.1)
    assert [round(b.reaction_s, 3) for b in barge_ins] == [0.4]


def test_one_bot_region_is_one_barge_in():
    barge_ins = find_barge_ins(regions((2, 2.5), (3, 4)), regions((0, 5)), min_overlap_s=0.1)
    assert [b.caller_start_s for b in barge_ins] == [2]


def test_silence_runs_through_the_end_of_the_call():
    assert find_silences(regions((0, 1)), [], end_s=31) == regions((1, 31))


def test_no_trailing_silence_when_speech_reaches_the_end():
    assert find_silences(regions((0, 2)), regions((2, 5)), end_s=5) == []


def test_a_call_with_no_speech_is_all_silence():
    assert find_silences([], [], end_s=4) == regions((0, 4))


def test_unanswered_turn_at_the_end_of_the_call():
    [item] = unanswered_turns(regions((0, 1)), [], end_s=31)
    assert (item.caller_end_s, item.waited_s) == (1, 30)


def test_unanswered_turn_before_the_caller_tries_again():
    [item] = unanswered_turns(regions((0, 1), (6, 7)), regions((8, 9)), end_s=10)
    assert (item.caller_end_s, item.waited_s) == (1, 5)


def test_answered_turns_are_not_unanswered():
    assert unanswered_turns(regions((0, 1)), regions((2, 3)), end_s=4) == []


def test_a_bot_already_talking_when_the_caller_stops_counts_as_an_answer():
    assert unanswered_turns(regions((0, 2)), regions((1, 4)), end_s=5) == []


async def test_a_bot_that_never_answers_fails_on_timing(store):
    out, _ = await evaluate(store, caller=[(0, 1)], duration_s=31.0)
    assert out.metrics["dead_air"]["longest_s"] == pytest.approx(30.0, abs=TOLERANCE)
    assert out.metrics["dead_air"]["count_over_threshold"] == 1
    assert [t["waited_s"] for t in out.metrics["unanswered_turns"]] == pytest.approx(
        [30.0], abs=TOLERANCE
    )
    unanswered = next(f for f in out.findings if f.category == "unanswered_turn")
    assert unanswered.severity == "major"
    assert unanswered.advisory is False
    assert "dead_air" in categories(out)


async def test_a_short_wait_after_a_goodbye_is_not_a_finding(store):
    out, _ = await evaluate(store, caller=[(0, 1)], bot=[(1.5, 3), (3.5, 5)], duration_s=5.0)
    assert out.metrics["unanswered_turns"] == []
    out, _ = await evaluate(store, bot=[(0, 3)], caller=[(3.5, 5)], duration_s=6.0)
    assert len(out.metrics["unanswered_turns"]) == 1
    assert "unanswered_turn" not in categories(out)


async def test_synthetic_timing_lists_unanswered_turns_without_judging_the_wait(store):
    artifact = CallArtifact(mode=CallMode.TURN_BASED, timing_is_real=False)
    out, _ = await evaluate(store, artifact, caller=[(0, 1)], duration_s=31.0)
    assert len(out.metrics["unanswered_turns"]) == 1
    assert out.findings == []


async def test_reaction_stops_at_the_speech_the_caller_cut_into(store):
    # With nobody in the bot's later pause its regions merge into one turn, and that turn would
    # report a 5 s reaction to an interruption the bot stopped for after 1 s.
    out, _ = await evaluate(store, bot=[(0, 2), (2.5, 6)], caller=[(1, 1.9)], duration_s=7.0)
    events = out.metrics["barge_in"]["events"]
    assert [e["reaction_s"] for e in events] == pytest.approx([1.0], abs=TOLERANCE)
