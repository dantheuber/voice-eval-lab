"""The caller bot and the text-only call runner."""

import json

import pytest
from scripted_llm import ScriptedLLM, call, caller_says, uses

from voicelab.agents.caller import (
    MAX_TURNS_LINE,
    REDIRECT_LINE,
    CallerBot,
    CallerLine,
    run_text_call,
)
from voicelab.agents.seeds import load_personas, load_scenarios
from voicelab.agents.support import SupportBot
from voicelab.company import ToolDispatcher, create_call_db
from voicelab.core.models import (
    BargeInEvent,
    Beat,
    CallArtifact,
    CallMode,
    Persona,
    Rule,
    Scenario,
    Speaker,
    SystemEvent,
    ToolCallEvent,
    Utterance,
    VoiceRef,
)


def persona(**fields) -> Persona:
    return Persona(name="Test", voice=VoiceRef(voice_id="default_female_1"), **fields)


def scenario(**fields) -> Scenario:
    defaults = {
        "id": "scn_test",
        "title": "Test",
        "goal": "You were charged twice.",
        "facts": {"name": "Dana", "pin": "4821", "phone": "(503) 555-0101", "invoice": "FH-4419"},
        "beats": [
            Beat(id="say_problem", instruction="Say you were charged twice."),
            Beat(id="give_pin", instruction="Give your PIN."),
        ],
    }
    return Scenario(**(defaults | fields))


def caller(*script, default=None, who=None, what=None, **options):
    llm = ScriptedLLM(*script, default=default)
    return CallerBot(llm, who or persona(), what or scenario(), **options), llm


def system_text(llm, index=-1):
    return llm.requests[index]["messages"][0].content


def request_text(llm, index=-1):
    return llm.requests[index]["messages"][1].content


# The prompt holds the caller to the sheet -------------------------------------------------------


def test_prompt_lists_the_facts_as_the_only_facts():
    bot, _ = caller()
    prompt = bot.system_prompt()
    assert "- pin: 4821" in prompt
    assert "- phone: (503) 555-0101" in prompt
    assert "only facts you know" in prompt
    assert "don't know" in prompt
    assert "Never guess or make up" in prompt


def test_prompt_leaves_out_what_the_caller_must_not_know():
    what = scenario(
        must=[Rule(text="SECRET-MUST")],
        must_not=[Rule(text="SECRET-MUST-NOT")],
        description="SECRET-DESCRIPTION",
    )
    bot, _ = caller(what=what)
    prompt = bot.system_prompt()
    assert "SECRET" not in prompt
    assert "Give your PIN" not in prompt  # later beats stay hidden until they're current


def test_prompt_says_the_caller_is_not_an_assistant():
    prompt = caller()[0].system_prompt()
    assert "not an assistant" in prompt
    assert "Never help the agent, summarize for it, or offer to do its job" in prompt


def test_prompt_asks_for_spoken_english():
    prompt = caller()[0].system_prompt()
    assert "Fragments and fillers" in prompt
    assert "No lists" in prompt
    assert "one or two sentences" in prompt


def test_prompt_with_an_empty_sheet_has_no_facts():
    bot, _ = caller(what=scenario(facts={}))
    assert "- (nothing)" in bot.system_prompt()


@pytest.mark.parametrize(
    ("fields", "phrase"),
    [
        ({"age": 78}, "78 years old"),
        ({"temperament": "angry"}, "Your mood: angry"),
        ({"patience": 1}, "no patience"),
        ({"patience": 5}, "very patient"),
        ({"verbosity": 1}, "as few words"),
        ({"verbosity": 5}, "ramble"),
        ({"tech_savviness": 1}, "the email thing"),
        ({"tech_savviness": 5}, "MX record, TTL, and IMAP"),
        ({"speaking_style": "Says oh dear a lot."}, "Says oh dear a lot."),
    ],
)
def test_persona_shows_up_as_words(fields, phrase):
    assert phrase in caller(who=persona(**fields))[0].system_prompt()


def test_persona_label_is_not_mistaken_for_the_callers_name():
    who = persona().model_copy(update={"name": "Impatient business owner"})
    assert "Impatient business owner" not in caller(who=who)[0].system_prompt()


# Conversation -----------------------------------------------------------------------------------


async def test_returns_the_line_and_records_what_was_heard():
    bot, llm = caller(caller_says("Yeah, um, I got charged twice."))
    line = await bot.next_line("Thanks for calling.")
    assert line == CallerLine("Yeah, um, I got charged twice.")
    assert "Agent: Thanks for calling." in request_text(llm)
    assert bot.transcript[-1] == ("You", "Yeah, um, I got charged twice.")


async def test_asks_for_structured_json():
    bot, llm = caller(caller_says("Hi."))
    await bot.next_line("Hello")
    schema = llm.requests[0]["json_schema"]
    assert set(schema["required"]) == {"say", "completed_beat", "hang_up", "hang_up_reason"}


async def test_earlier_lines_stay_in_the_prompt():
    bot, llm = caller(caller_says("First."), caller_says("Second."))
    await bot.next_line("Hello")
    await bot.next_line("Go on")
    text = request_text(llm)
    assert text.index("Agent: Hello") < text.index("You: First.") < text.index("Agent: Go on")


async def test_transcript_in_the_prompt_is_bounded():
    bot, llm = caller(default=caller_says("Still here."))
    bot.scenario.max_turns = 50
    bot.persona.patience = 5
    for number in range(8):
        await bot.next_line(f"Bot line {number}")
    bot.transcript.extend(("Agent", f"old {n}") for n in range(30))
    await bot.next_line("Latest")
    assert "Bot line 0" not in request_text(llm)
    assert "Latest" in request_text(llm)


# Beats ------------------------------------------------------------------------------------------


async def test_the_current_beat_is_in_the_prompt_and_moves_on():
    bot, llm = caller(
        caller_says("Charged twice.", "say_problem"),
        caller_says("4821.", "give_pin"),
    )
    assert bot.current_beat.id == "say_problem"
    line = await bot.next_line("Hi")
    assert line.completed_beat == "say_problem"
    assert "Say you were charged twice." in system_text(llm, 0)
    assert bot.current_beat.id == "give_pin"
    await bot.next_line("PIN please")
    assert "Give your PIN." in system_text(llm, 1)
    assert "Say you were charged twice." not in system_text(llm, 1)
    assert bot.completed_beats == ["say_problem", "give_pin"]


async def test_derailed_conversation_stays_on_the_beat():
    bot, llm = caller(default=caller_says("Yeah, no, about my charge."))
    await bot.next_line("Have you heard about our new domain sale?")
    await bot.next_line("Let's talk about website builders instead.")
    assert bot.current_beat.id == "say_problem"
    assert "Say you were charged twice." in system_text(llm)
    assert "Bring it back to your current step" in system_text(llm)


async def test_a_later_beat_id_does_not_skip_ahead():
    bot, _ = caller(caller_says("Sure.", "give_pin"))
    line = await bot.next_line("Hi")
    assert line.completed_beat is None
    assert bot.current_beat.id == "say_problem"
    assert bot.completed_beats == []


async def test_a_made_up_beat_id_is_ignored():
    bot, _ = caller(caller_says("Sure.", "null"))
    assert (await bot.next_line("Hi")).completed_beat is None


async def test_schema_only_allows_the_current_beat():
    bot, llm = caller(caller_says("Charged.", "say_problem"), caller_says("Ok."))
    await bot.next_line("Hi")
    await bot.next_line("Yes")
    first, second = [request["json_schema"] for request in llm.requests]
    assert first["properties"]["completed_beat"]["enum"] == ["say_problem", None]
    assert second["properties"]["completed_beat"]["enum"] == ["give_pin", None]


async def test_optional_beat_is_dropped_when_it_goes_nowhere():
    what = scenario(
        beats=[
            Beat(id="maybe", instruction="Mention the weather.", required=False),
            Beat(id="must", instruction="Ask for a refund."),
        ]
    )
    bot, llm = caller(default=caller_says("Hm."), what=what)
    await bot.next_line("Hi")
    assert bot.current_beat.id == "maybe"
    await bot.next_line("Hi")
    assert bot.current_beat.id == "must"
    assert bot.completed_beats == []
    await bot.next_line("Hi")
    assert "Ask for a refund." in system_text(llm)


async def test_a_required_beat_is_never_dropped():
    bot, _ = caller(default=caller_says("Hm."))
    for _ in range(4):
        await bot.next_line("Hi")
    assert bot.current_beat.id == "say_problem"


# Hanging up -------------------------------------------------------------------------------------


def no_script():
    return scenario(beats=[])


async def test_reports_a_hang_up_and_why():
    bot, _ = caller(
        caller_says("Forget it.", hang_up=True, reason="the agent refused"), what=no_script()
    )
    line = await bot.next_line("No.")
    assert line.hang_up is True
    assert line.hang_up_reason == "the agent refused"
    assert line.text == "Forget it."
    assert bot.hung_up is True
    assert bot.hang_up_reason == "the agent refused"


async def test_hang_up_without_a_reason_still_has_one():
    bot, _ = caller(caller_says("Bye.", hang_up=True), what=no_script())
    assert (await bot.next_line("No.")).hang_up_reason


async def test_a_beat_can_complete_on_the_hang_up_turn():
    what = scenario(beats=[Beat(id="only", instruction="Say goodbye.")])
    bot, _ = caller(caller_says("Bye.", "only", hang_up=True, reason="done"), what=what)
    line = await bot.next_line("Thanks.")
    assert (line.completed_beat, line.hang_up) == ("only", True)
    assert bot.finished_beats


async def test_silent_hang_up_is_allowed():
    bot, _ = caller(caller_says("", hang_up=True, reason="gave up"), what=no_script())
    assert (await bot.next_line("Hello")).text == ""


async def test_it_stays_on_the_line_until_the_script_is_done():
    bot, llm = caller(
        caller_says("Thanks, bye.", hang_up=True, reason="all sorted"),
        caller_says("Thanks, bye."),
        caller_says("I was charged twice.", "say_problem"),
    )
    line = await bot.next_line("Hello?")
    assert (line.text, line.hang_up) == ("I was charged twice.", False)
    assert "don't say goodbye" in request_text(llm)
    assert "Say you were charged twice." in request_text(llm)


async def test_a_caller_that_wont_stay_gets_the_safe_line_and_stays():
    bot, _ = caller(default=caller_says("Bye.", hang_up=True, reason="done"))
    line = await bot.next_line("Hello?")
    assert (line.text, line.hang_up) == (REDIRECT_LINE, False)


async def test_out_of_patience_it_can_hang_up_with_beats_left():
    impatient = persona(patience=1)
    bot, _ = caller(
        *[caller_says("Hm.") for _ in range(4)],
        caller_says("Forget it, bye.", hang_up=True, reason="fed up"),
        who=impatient,
    )
    for _ in range(4):
        assert not (await bot.next_line("Sorry?")).hang_up
    line = await bot.next_line("Sorry?")
    assert (line.hang_up, line.hang_up_reason) == (True, "fed up")


async def test_it_waits_for_the_answer_to_its_last_question():
    what = scenario(beats=[Beat(id="only", instruction="Ask how long it takes.")])
    bot, _ = caller(
        caller_says("How long will it take?", "only", hang_up=True, reason="done"),
        caller_says("Great, thanks."),
        what=what,
    )
    assert not (await bot.next_line("All set.")).hang_up
    last = await bot.next_line("About an hour.")
    assert (last.hang_up, last.hang_up_reason) == (True, "finished")


async def test_an_optional_last_beat_doesnt_keep_it_on_the_line():
    what = scenario(
        beats=[
            Beat(id="ask", instruction="Ask for a refund."),
            Beat(id="close", instruction="Say thanks.", required=False),
        ]
    )
    bot, _ = caller(
        caller_says("Refund please.", "ask"),
        caller_says("Thanks, bye.", hang_up=True, reason="done"),
        what=what,
    )
    await bot.next_line("Hi")
    assert (await bot.next_line("Done.")).hang_up


@pytest.mark.parametrize(
    "say", ["The agent said it was done.", "That isn't on my sheet.", "My step says to wait."]
)
async def test_the_script_never_shows(say):
    bot, llm = caller(caller_says(say), caller_says("I was charged twice."))
    assert (await bot.next_line("Hi")).text == "I was charged twice."
    assert "never mention a sheet" in request_text(llm)


async def test_a_pin_beat_waits_for_the_zip_and_pin():
    what = scenario(
        facts={"billing_zip": "97214", "pin": "4821"},
        beats=[Beat(id="identify", instruction="Give your billing zip and PIN when asked.")],
    )
    bot, _ = caller(
        caller_says("My zip is 97214.", "identify"),
        caller_says("And the pin is four, eight, two, one.", "identify"),
        what=what,
    )
    assert (await bot.next_line("Zip and PIN?")).completed_beat is None
    assert (await bot.next_line("And the PIN?")).completed_beat == "identify"


async def test_a_beat_that_asks_for_a_yes_waits_for_one():
    what = scenario(beats=[Beat(id="agree", instruction="Ask the price, then say yes.")])
    bot, _ = caller(
        caller_says("How much is it?", "agree"),
        caller_says("Yes, go ahead.", "agree"),
        what=what,
    )
    assert (await bot.next_line("Shall I renew it?")).completed_beat is None
    assert (await bot.next_line("It's $14.99.")).completed_beat == "agree"


async def test_a_beat_it_forgets_to_claim_still_shows_in_its_words():
    what = scenario(
        facts={"billing_zip": "97214", "pin": "4821"},
        beats=[
            Beat(id="identify", instruction="Give your billing zip and PIN when asked."),
            Beat(id="agree", instruction="Say yes to the charge."),
        ],
    )
    bot, _ = caller(
        caller_says("It's 97214, and the pin is 4821."),
        caller_says("Yes, go ahead."),
        what=what,
    )
    assert (await bot.next_line("Zip and PIN?")).completed_beat == "identify"
    assert (await bot.next_line("Shall I charge it?")).completed_beat == "agree"


async def test_a_yes_to_nothing_doesnt_finish_a_beat():
    what = scenario(beats=[Beat(id="agree", instruction="Say yes to the charge.")])
    bot, _ = caller(caller_says("Yes, hello."), what=what)
    assert (await bot.next_line("Thanks for holding.")).completed_beat is None


async def test_a_question_after_the_last_beat_gets_its_answer_then_the_call_ends():
    what = scenario(beats=[Beat(id="only", instruction="Ask how long it takes.")])
    bot, _ = caller(
        caller_says("How long?", "only"),
        caller_says("Sorry, how long?"),
        caller_says("Sorry, what?"),
        caller_says("And what?"),
        what=what,
    )
    await bot.next_line("All set.")
    assert not (await bot.next_line("D N S changes usually")).hang_up
    assert not (await bot.next_line("About")).hang_up
    assert (await bot.next_line("An hour.")).hang_up


async def test_a_long_winded_line_is_cut_at_a_sentence():
    chatty = persona(verbosity=5)
    limit = 18 + 6 * 5
    sentence = " ".join(["word"] * 20) + "."
    bot, _ = caller(caller_says(f"{sentence} {sentence} {sentence}"), who=chatty)
    line = await bot.next_line("Hi")
    assert line.text == f"{sentence} {sentence}"
    assert len(line.text.split()) <= limit


async def test_after_the_last_beat_it_says_goodbye_and_hangs_up():
    what = scenario(beats=[Beat(id="only", instruction="Ask for a refund.")])
    bot, llm = caller(caller_says("Refund please.", "only"), caller_says("Thanks, bye."), what=what)
    first = await bot.next_line("Hi")
    assert first.hang_up is False
    second = await bot.next_line("Done.")
    assert "Say a short goodbye and hang up" in system_text(llm)
    assert second.hang_up is True
    assert second.hang_up_reason == "finished"


async def test_max_turns_ends_cleanly_without_asking_the_llm():
    bot, llm = caller(caller_says("One."), caller_says("Two."), what=scenario(max_turns=2))
    await bot.next_line("a")
    await bot.next_line("b")
    line = await bot.next_line("c")
    assert line.text == MAX_TURNS_LINE
    assert (line.hang_up, line.hang_up_reason) == (True, "max_turns")
    assert len(llm.requests) == 2


async def test_nothing_happens_after_hanging_up():
    bot, llm = caller(caller_says("Bye.", hang_up=True, reason="done"), what=no_script())
    await bot.next_line("Hi")
    line = await bot.next_line("Hello?")
    assert line == CallerLine("", hang_up=True, hang_up_reason="done")
    assert len(llm.requests) == 1


# Patience ---------------------------------------------------------------------------------------


async def test_impatient_caller_gets_annoyed_then_gives_up():
    who = persona(patience=1)
    bot, llm = caller(default=caller_says("What?"), who=who)
    assert bot.patience_limit == 5
    lines = [await bot.next_line("Please hold.") for _ in range(5)]
    assert [line.hang_up for line in lines] == [False, False, False, False, True]
    assert lines[-1].hang_up_reason == "out of patience"
    assert "getting annoyed" not in system_text(llm, 0)
    assert "getting annoyed" in system_text(llm, 2)
    assert "had enough" in system_text(llm, 4)


async def test_patient_caller_lasts_longer():
    bot, _ = caller(default=caller_says("Okay."), who=persona(patience=5), what=scenario())
    lines = [await bot.next_line("Please hold.") for _ in range(12)]
    assert not any(line.hang_up for line in lines)
    assert (await bot.next_line("Please hold.")).hang_up_reason == "out of patience"


async def test_progress_resets_patience():
    bot, llm = caller(
        caller_says("Hm."),
        caller_says("Hm."),
        caller_says("Charged.", "say_problem"),
        caller_says("Hm."),
        who=persona(patience=1),
    )
    for _ in range(4):
        await bot.next_line("Hi")
    assert "getting annoyed" not in system_text(llm)


# Guards -----------------------------------------------------------------------------------------


async def test_invented_digits_are_rejected_and_retried():
    bot, llm = caller(caller_says("My account number is 998877."), caller_says("I don't have it."))
    line = await bot.next_line("What's your account number?")
    assert line.text == "I don't have it."
    assert len(llm.requests) == 2
    assert "998877" in request_text(llm)
    assert "not on your fact sheet" in request_text(llm)


async def test_invented_email_is_rejected():
    bot, llm = caller(caller_says("It's dana@madeup.example."), caller_says("Not sure."))
    assert (await bot.next_line("Email?")).text == "Not sure."
    assert "dana@madeup.example" in request_text(llm)


async def test_repeat_offender_falls_back_to_a_safe_line():
    bot, _ = caller(default=caller_says("My PIN is 9999."))
    assert (await bot.next_line("PIN?")).text == REDIRECT_LINE


@pytest.mark.parametrize(
    "say",
    [
        "My PIN is 4821.",
        "My PIN is 4 8 2 1.",
        "It's 503-555-0101.",
        "It's (503) 555-0101.",
        "503 555 0101",
        "Invoice FH-4419.",
        "I got charged 2 or 3 times.",
        "4821 503",
        "It happened 10 days ago.",
    ],
)
async def test_sheet_facts_and_small_numbers_pass_the_guard(say):
    bot, llm = caller(caller_says(say))
    assert (await bot.next_line("Go ahead.")).text == say
    assert len(llm.requests) == 1


async def test_repeating_what_the_agent_said_is_allowed():
    bot, llm = caller(caller_says("Ticket 2203, got it."))
    await bot.next_line("Your ticket is 2203.")
    assert len(llm.requests) == 1


async def test_spaced_digits_are_treated_as_one_number():
    bot, llm = caller(caller_says("It's 7 7 7 7."), caller_says("Don't know."))
    assert (await bot.next_line("PIN?")).text == "Don't know."
    assert "7 7 7 7" in request_text(llm)


SHEET = {"name": "Renee", "billing_zip": "92103", "pin": "1234", "phone": "(619) 555-0199"}


def failed_verification_caller(*script):
    return caller(*script, what=scenario(facts=SHEET))


async def test_a_disclosed_pin_cannot_replace_the_sheet_on_the_seeded_scenario():
    what = load_scenarios()["scn_failed_verification"]
    bot, llm = caller(caller_says("My PIN is 3197."), caller_says("It's 1234."), what=what)
    assert (await bot.next_line("Is your PIN 3197?")).text == "It's 1234."
    assert "3197" in request_text(llm).split("rejected")[1]


@pytest.mark.parametrize(
    ("heard", "say"),
    [
        ("Is your PIN 3197?", "My PIN is 3197."),
        ("Is your PIN 3197?", "3197."),
        ("Is your PIN 3197?", "It's 3 1 9 7."),
        ("Is your PIN 3197?", "My PIN is three one nine seven."),
        ("Is your PIN 3197?", "Um, three, one, nine, seven."),
        ("Is it 3197?", "3197."),
        ("Your billing zip is 92101, right?", "Yeah, 92101."),
        ("Is your phone (619) 555-0106?", "Yes, (619) 555-0106."),
        ("Your card ends in 2290.", "Right, 2290."),
        ("Is your email renee@castillostudio.example?", "Yes, renee@castillostudio.example."),
        ("Your ticket is 3197.", "My PIN is 3197."),
        ("Your invoice went to renee@other.example.", "Yes, renee@other.example."),
    ],
)
async def test_disclosed_identity_details_are_not_the_callers_to_state(heard, say):
    bot, llm = failed_verification_caller(caller_says(say), caller_says("I don't have that."))
    assert (await bot.next_line(heard)).text == "I don't have that."
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    ("heard", "say"),
    [
        ("Your ticket is 2203.", "Ticket 2203, got it."),
        ("Your ticket is T K dash 2 2 0 3.", "So TK-2203?"),
        ("Your invoice is F H dash 4 4 1 9.", "Invoice FH-4419, okay."),
        ("Your confirmation number is 88231.", "88231, thanks."),
        ("The renewal is $114.99.", "$114.99, okay."),
        ("Use port 993.", "993?"),
        ("It expires in 2027.", "Oh, 2027."),
    ],
)
async def test_references_learned_on_the_call_can_be_repeated(heard, say):
    bot, llm = failed_verification_caller(caller_says(say))
    assert (await bot.next_line(heard)).text == say
    assert len(llm.requests) == 1


@pytest.mark.parametrize(
    "say",
    ["My PIN is 1234.", "No, it's 1234.", "Zip is 92103.", "No, it's (619) 555-0199."],
)
async def test_the_sheet_still_wins_after_a_disclosure(say):
    bot, llm = failed_verification_caller(caller_says(say))
    assert (await bot.next_line("Is your PIN 3197?")).text == say
    assert len(llm.requests) == 1


async def test_only_the_agents_lines_can_teach_a_reference():
    bot, llm = failed_verification_caller(*[caller_says("Ticket 777888, I think.")] * 3)
    assert (await bot.next_line("Your ticket is on its way.")).text == REDIRECT_LINE
    assert len(llm.requests) == 3


@pytest.mark.parametrize(
    "say", ["It's nine nine eight eight.", "Um, seven oh seven seven.", "Zero zero one two."]
)
async def test_invented_numbers_in_words_are_rejected(say):
    bot, llm = caller(caller_says(say), caller_says("I don't know."))
    assert (await bot.next_line("PIN?")).text == "I don't know."


@pytest.mark.parametrize("say", ["four eight two one", "It's Four, Eight, Two, One."])
async def test_sheet_numbers_in_words_pass_the_guard(say):
    bot, llm = caller(caller_says(say))
    assert (await bot.next_line("PIN?")).text == say
    assert len(llm.requests) == 1


def test_prompt_says_the_agent_cannot_change_the_sheet():
    prompt = caller()[0].system_prompt()
    assert "your sheet is right" in prompt


@pytest.mark.parametrize(
    "say",
    [
        "How can I help you today?",
        "Is there anything else I can do for you?",
        "I'd be happy to help with that.",
        "Let me know if you need anything.",
    ],
)
async def test_sounding_like_an_agent_is_rejected(say):
    bot, llm = caller(caller_says(say), caller_says("Yeah, my charge."))
    assert (await bot.next_line("Hello?")).text == "Yeah, my charge."
    assert "You are the customer" in request_text(llm)


# Bad LLM output ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["not json", "", '{"say": ""}', '{"say": 5}', "[1, 2]", '{"say": "unterminated'],
)
async def test_bad_json_is_retried(bad):
    bot, llm = caller(bad, caller_says("Hello?"))
    assert (await bot.next_line("Hi")).text == "Hello?"
    assert "not valid JSON" in request_text(llm)


@pytest.mark.parametrize("bad", ['{"say": ' + "9" * 5000 + "}", '{"say": ' + "[" * 200000 + "}"])
async def test_json_python_cannot_parse_is_retried_not_a_crash(bad):
    bot, llm = caller(bad, caller_says("Hello?"))
    assert (await bot.next_line("Hi")).text == "Hello?"
    assert len(llm.requests) == 2


async def test_lone_surrogates_never_reach_the_transcript():
    bot, _ = caller(json.dumps({"say": "Hi \ud800there", "completed_beat": None}))
    line = await bot.next_line("Hello")
    line.text.encode("utf-8")


async def test_bad_json_twice_falls_back():
    bot, _ = caller(default="nope")
    assert (await bot.next_line("Hi")).text == REDIRECT_LINE


async def test_json_inside_a_code_fence_is_accepted():
    bot, llm = caller(f"```json\n{caller_says('Hi there.')}\n```")
    assert (await bot.next_line("Hi")).text == "Hi there."
    assert len(llm.requests) == 1


# Interruptions ----------------------------------------------------------------------------------


def test_interruption_is_deterministic_for_a_seed():
    who = persona(interruption_rate=0.5)
    first, _ = caller(who=who, seed=7)
    second, _ = caller(who=who, seed=7)
    assert [first.decide_interruption(n) for n in range(30)] == [
        second.decide_interruption(n) for n in range(30)
    ]


def test_asking_twice_gives_the_same_answer():
    bot, _ = caller(who=persona(interruption_rate=0.5))
    assert bot.decide_interruption(3) == bot.decide_interruption(3)


def test_seeds_change_the_decisions():
    who = persona(interruption_rate=0.5)
    first, _ = caller(who=who, seed=1)
    second, _ = caller(who=who, seed=2)
    decisions = [
        (first.decide_interruption(n).interrupt, second.decide_interruption(n).interrupt)
        for n in range(30)
    ]
    assert any(one != two for one, two in decisions)


@pytest.mark.parametrize(("rate", "expected"), [(0.0, False), (1.0, True)])
def test_extreme_rates(rate, expected):
    bot, _ = caller(who=persona(interruption_rate=rate))
    assert all(bot.decide_interruption(n).interrupt is expected for n in range(50))


def test_rate_is_roughly_honored():
    bot, _ = caller(who=persona(interruption_rate=0.3))
    hits = sum(bot.decide_interruption(n).interrupt for n in range(2000))
    assert 500 < hits < 700


def test_cut_in_point_lands_mid_speech():
    bot, _ = caller(who=persona(interruption_rate=1.0))
    for number in range(50):
        assert 0.2 <= bot.decide_interruption(number).after_fraction <= 0.7


# Text calls -------------------------------------------------------------------------------------


@pytest.fixture
def db():
    call_db = create_call_db()
    yield call_db
    call_db.close()


def support(db, *script, **options):
    return SupportBot(ScriptedLLM(*script), ToolDispatcher(db), **options)


def utterances(result, speaker=None):
    found = [event for event in result.timeline if isinstance(event, Utterance)]
    return [event for event in found if speaker is None or event.speaker == speaker]


def kinds(result):
    return [event.kind for event in result.timeline if isinstance(event, SystemEvent)]


def free_caller(*script, **options):
    """A caller with no beats, so a test can hang it up whenever it likes."""
    return caller(*script, what=options.pop("what", no_script()), **options)


async def test_text_call_runs_both_bots_to_the_end(db):
    bot = support(
        db,
        "Sure. Can I get your phone number?",
        uses(call("find_customer", phone="(503) 555-0101")),
        "Thanks Dana, invoice FH-4419 was $49.99. What now?",
        "Thanks for that.",
    )
    who, _ = caller(
        caller_says("I got charged twice.", "say_problem"),
        caller_says("(503) 555-0101."),
        caller_says("My PIN is 4821.", "give_pin"),
        caller_says("Okay bye.", hang_up=True, reason="done"),
    )
    result = await run_text_call(bot, who)

    assert result.end_reason == "caller_hung_up"
    assert result.detail == "done"
    assert result.completed_beats == ["say_problem", "give_pin"]
    assert kinds(result)[0] == "call_started"
    assert kinds(result)[-2:] == ["caller_hung_up", "call_ended"]
    assert kinds(result).count("beat_completed") == 2

    tools = [event for event in result.timeline if isinstance(event, ToolCallEvent)]
    assert [event.name for event in tools] == ["find_customer"]


async def test_text_call_events_are_ordered_and_timed(db):
    bot = support(db, uses(call("list_plans")), "We have plans.")
    who, _ = free_caller(
        caller_says("What plans?"),
        caller_says("Bye.", hang_up=True, reason="done"),
    )
    result = await run_text_call(bot, who)
    starts = [
        event.start_s if isinstance(event, Utterance) else event.at_s for event in result.timeline
    ]
    assert starts == sorted(starts)
    for utterance in utterances(result):
        assert utterance.end_s > utterance.start_s


async def test_timeline_fits_the_call_artifact(db):
    bot = support(db, uses(call("list_plans")), "We have plans.")
    who, _ = free_caller(
        caller_says("What plans?"),
        caller_says("Bye.", hang_up=True, reason="done"),
        what=no_script(),
    )
    result = await run_text_call(bot, who)
    artifact = CallArtifact(
        mode=CallMode.TURN_BASED,
        timing_is_real=False,
        timeline=result.timeline,
        db_changes=result.db_changes,
    )
    assert CallArtifact.model_validate_json(artifact.model_dump_json()) == artifact


async def test_caller_hears_the_spoken_form(db):
    bot = support(db, "Your invoice is FH-4419.")
    who, llm = free_caller(
        caller_says("Which invoice?"), caller_says("Bye.", hang_up=True, reason="x")
    )
    await run_text_call(bot, who)
    assert "F, as in frank" in request_text(llm)
    assert "FH-4419." not in request_text(llm).split("Reply with JSON")[0].split("Agent:")[-1]


async def test_utterances_record_intended_spoken_and_heard(db):
    bot = support(db, "Okay then.")
    who, _ = free_caller(caller_says("Hi."), caller_says("Bye.", hang_up=True, reason="x"))
    result = await run_text_call(bot, who)
    first_bot = utterances(result, Speaker.BOT)[0]
    assert first_bot.intended_text == first_bot.spoken_text == first_bot.heard_text
    assert utterances(result, Speaker.CALLER)[0].intended_text == "Hi."


async def test_bot_can_end_the_call(db):
    bot = support(db, uses(call("end_call", reason="all done"), text="Goodbye."))
    who, llm = free_caller(caller_says("That's all, thanks."))
    result = await run_text_call(bot, who)
    assert (result.end_reason, result.detail) == ("bot_ended", "all done")
    assert kinds(result)[-2:] == ["bot_ended", "call_ended"]
    assert len(llm.requests) == 1


async def test_call_stops_at_max_turns(db):
    what = scenario(max_turns=3)
    bot = SupportBot(ScriptedLLM(default="Please hold."), ToolDispatcher(db))
    who, llm = free_caller(default=caller_says("Hello?"), what=what, who=persona(patience=5))
    result = await run_text_call(bot, who)
    assert result.end_reason == "max_turns"
    assert kinds(result)[-2:] == ["max_turns", "call_ended"]
    assert len(llm.requests) == 3
    assert utterances(result, Speaker.CALLER)[-1].intended_text == MAX_TURNS_LINE


async def test_db_changes_come_back_with_the_result(db):
    bot = support(db, uses(call("unlock_mailbox", address="evelyn@hartfamily.example")), "All set.")
    who, _ = free_caller(caller_says("Unlock it."), caller_says("Bye.", hang_up=True, reason="x"))
    result = await run_text_call(bot, who)
    assert [(change.table, change.kind) for change in result.db_changes] == [
        ("mailboxes", "update")
    ]


async def test_barge_in_trims_what_the_caller_hears_and_the_bots_memory(db):
    long_reply = "One two three four five six seven eight nine ten."
    llm = ScriptedLLM(long_reply, default="Okay.")
    bot = SupportBot(llm, ToolDispatcher(db))
    who, caller_llm = free_caller(
        caller_says("Go."),
        caller_says("Stop."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=persona(interruption_rate=1.0, patience=5),
    )
    result = await run_text_call(bot, who)

    barge_ins = [event for event in result.timeline if isinstance(event, BargeInEvent)]
    assert len(barge_ins) == 2  # the greeting and the long reply; "Okay." has nothing to cut
    interrupted = [event for event in utterances(result, Speaker.BOT) if event.interrupted]
    target = interrupted[1]
    assert target.intended_text == long_reply
    assert target.spoken_text != long_reply
    assert long_reply.startswith(target.spoken_text)
    assert target.heard_text == target.spoken_text
    assert barge_ins[1].interrupted_utterance_id == target.id

    heard_by_caller = "".join(m.content for m in caller_llm.requests[-1]["messages"])
    assert "ten." not in heard_by_caller
    assert long_reply not in " ".join(m.content for m in bot.messages)
    assert target.spoken_text in " ".join(m.content for m in bot.messages)


LONG_LINE = " ".join(["word"] * 20) + "."  # longer than any cut-in point


def always_interrupts(**fields):
    return persona(interruption_rate=1.0, patience=5, **fields)


async def test_tools_in_later_rounds_never_run_after_the_cut(db):
    llm = ScriptedLLM(
        uses(call("list_plans")),
        uses(call("unlock_mailbox", address="evelyn@hartfamily.example")),
        "It's unlocked.",
        default="Okay.",
    )
    bot = SupportBot(llm, ToolDispatcher(db), filler_after_s=0)
    who, _ = free_caller(
        caller_says("Go."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=always_interrupts(),
        seed=3,
    )
    result = await run_text_call(bot, who)

    assert result.db_changes == []
    assert [e for e in result.timeline if isinstance(e, ToolCallEvent)] == []
    assert len(llm.requests) == 1
    assert llm.closed_streams == 1


async def test_the_llm_stream_closes_before_the_caller_answers(db):
    class WatchingLLM(ScriptedLLM):
        def __init__(self, watched, *script):
            super().__init__(*script)
            self.watched = watched
            self.closed_when_asked = []

        async def chat(self, messages, **options):
            self.closed_when_asked.append(self.watched.closed_streams)
            return await super().chat(messages, **options)

    support_llm = ScriptedLLM(f"{LONG_LINE} {LONG_LINE}")
    caller_llm = WatchingLLM(
        support_llm, caller_says("Go."), caller_says("Bye.", hang_up=True, reason="x")
    )
    bot = SupportBot(support_llm, ToolDispatcher(db))
    who = CallerBot(caller_llm, always_interrupts(), no_script())
    await run_text_call(bot, who)
    assert caller_llm.closed_when_asked == [0, 1]


async def test_tools_finished_before_the_cut_are_kept(db):
    llm = ScriptedLLM(
        uses(call("unlock_mailbox", address="evelyn@hartfamily.example")),
        LONG_LINE,
        default="Okay.",
    )
    bot = SupportBot(llm, ToolDispatcher(db))
    who, _ = free_caller(
        caller_says("Go."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=always_interrupts(),
    )
    result = await run_text_call(bot, who)
    assert [change.table for change in result.db_changes] == ["mailboxes"]
    assert utterances(result, Speaker.BOT)[1].interrupted is True


async def test_unplayed_sentences_are_not_intended_speech(db):
    sentences = ["Word " * 30 + "end.", "This later sentence never played."]
    bot = SupportBot(ScriptedLLM(" ".join(sentences), default="Okay."), ToolDispatcher(db))
    who, _ = free_caller(
        caller_says("Go."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=always_interrupts(),
    )
    result = await run_text_call(bot, who)
    cut = utterances(result, Speaker.BOT)[1]
    assert cut.interrupted is True
    assert "never played" not in cut.intended_text
    assert "never played" not in " ".join(m.content for m in bot.messages)


async def test_a_short_reply_can_still_be_cut_off(db):
    bot = SupportBot(ScriptedLLM(default="One two three four."), ToolDispatcher(db))
    who, _ = free_caller(
        caller_says("Go."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=always_interrupts(),
    )
    result = await run_text_call(bot, who)
    cut = utterances(result, Speaker.BOT)[1]
    assert cut.interrupted is True
    assert cut.spoken_text != cut.intended_text


async def test_a_one_word_reply_has_nothing_to_cut(db):
    bot = SupportBot(ScriptedLLM(default="Okay."), ToolDispatcher(db))
    who, _ = free_caller(
        caller_says("Go."),
        caller_says("Bye.", hang_up=True, reason="x"),
        who=always_interrupts(),
    )
    result = await run_text_call(bot, who)
    assert utterances(result, Speaker.BOT)[1].interrupted is False


async def test_the_final_goodbye_after_end_call_is_never_cut_off(db):
    bot = support(db, uses(call("end_call")))
    who, _ = free_caller(caller_says("Bye."), who=always_interrupts())
    result = await run_text_call(bot, who)
    assert result.end_reason == "bot_ended"
    assert utterances(result, Speaker.BOT)[-1].interrupted is False


async def test_cutting_in_before_end_call_ran_keeps_the_call_open(db):
    goodbye = "Well thank you so much for calling us today and have a really great rest of it."
    llm = ScriptedLLM(uses(call("end_call"), text=goodbye), default="Okay.")
    bot = SupportBot(llm, ToolDispatcher(db))
    who, _ = free_caller(
        caller_says("Bye."),
        caller_says("Wait, one more thing.", hang_up=True, reason="x"),
        who=always_interrupts(),
    )
    result = await run_text_call(bot, who)
    assert bot.ended is False
    assert result.end_reason == "caller_hung_up"
    assert [e for e in result.timeline if isinstance(e, ToolCallEvent)] == []


async def test_same_seed_same_call():
    async def run():
        database = create_call_db()
        bot = SupportBot(ScriptedLLM(default="Noted, please go on."), ToolDispatcher(database))
        who, _ = free_caller(
            default=caller_says("Hm."),
            who=persona(interruption_rate=0.5, patience=2),
            seed=11,
        )
        result = await run_text_call(bot, who)
        database.close()
        return [
            (event.speaker, event.spoken_text, event.interrupted)
            for event in result.timeline
            if isinstance(event, Utterance)
        ]

    assert await run() == await run()


async def test_text_call_works_with_a_seeded_scenario(db):
    what = load_scenarios()["scn_locked_mailbox"]
    who = load_personas()["per_confused_retiree"]
    bot = SupportBot(ScriptedLLM(default="Could you tell me more?"), ToolDispatcher(db))
    line, _ = caller(default=caller_says("Oh dear."), who=who, what=what)
    result = await run_text_call(bot, line)
    assert result.end_reason in {"caller_hung_up", "max_turns"}
