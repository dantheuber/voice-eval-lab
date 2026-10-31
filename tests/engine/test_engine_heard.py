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
        ("F, as in frank. H, as in house. Four, four, one, nine.", "FH4419."),
        ("F as in Frank H as in Henry 4419", "FH4419"),
        ("It's A, as in apple. C, as in cat. Nine, zero, seven, seven.", "It's AC9077."),
        ("H for Henry, 4 4 1 9", "H4419"),
        ("four one four, five five five, oh one one two", "4145550112"),
        ("dana at whitfieldlandscaping dot example", "dana@whitfieldlandscaping.example"),
        ("bellhardware dot example", "bellhardware.example"),
    ],
)
def test_spelled_ids_are_put_back_together(heard, repaired):
    assert prepare_heard_text(heard) == repaired


def test_the_anchor_word_outranks_a_misheard_letter():
    assert prepare_heard_text("S as in Frank, H as in house, 4 4 1 9") == "FH4419"


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
        "I was charged twice. As in two times.",
        "I have a question for Frank",
        "",
    ],
)
def test_ordinary_speech_is_left_alone(ordinary):
    assert prepare_heard_text(ordinary) == ordinary
