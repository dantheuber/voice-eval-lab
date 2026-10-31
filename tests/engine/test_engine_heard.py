import pytest

from voicelab.engine.heard import prepare_heard_text


@pytest.mark.parametrize(
    ("heard", "repaired"),
    [
        ("my account is a C nine zero seven seven", "my account is AC9077"),
        ("It's A C 9 0 7 7.", "It's AC9077."),
        ("invoice F H four four one nine please", "invoice FH4419 please"),
        ("my pin is four eight two one", "my pin is 4821"),
        ("oh two six four", "0264"),
    ],
)
def test_spelled_ids_are_put_back_together(heard, repaired):
    assert prepare_heard_text(heard) == repaired


@pytest.mark.parametrize(
    "ordinary",
    [
        "I have one or two questions",
        "a five digit code",
        "I was charged two times",
        "two hundred dollars",
        "I need a hand with this",
        "It was the 18th of October",
        "AC9077",
        "97214",
        "",
    ],
)
def test_ordinary_speech_is_left_alone(ordinary):
    assert prepare_heard_text(ordinary) == ordinary
