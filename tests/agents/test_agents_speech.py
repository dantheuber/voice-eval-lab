"""Speech normalization and sentence splitting."""

import pytest

from voicelab.agents.speech import (
    Lexicon,
    clean_markup,
    load_lexicon,
    normalize,
    number_to_words,
    ordinal_to_words,
    split_sentences,
    year_to_words,
)


def spoken(text: str) -> str:
    return normalize(text).spoken


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FH-4419", "F, as in frank. H, as in house. Four, four, one, nine."),
        ("$49.99", "forty nine dollars and ninety nine cents"),
        ("mail.fernhill.example", "mail dot fernhill dot example"),
    ],
)
def test_examples_from_the_brief(text, expected):
    assert spoken(text) == expected


def test_returns_display_and_spoken():
    result = normalize("Invoice FH-4419 is $49.99.")
    assert result.display == "Invoice FH-4419 is $49.99."
    assert result.spoken == (
        "Invoice F, as in frank. H, as in house. Four, four, one, nine. "
        "is forty nine dollars and ninety nine cents."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TK-2203", "T, as in tom. K, as in kite. Two, two, zero, three."),
        ("fh-4419", "F, as in frank. H, as in house. Four, four, one, nine."),
        ("FH4419", "F, as in frank. H, as in house. Four, four, one, nine."),
        ("AC9077", "A, as in apple. C, as in cat. Nine, zero, seven, seven."),
        (
            "Your invoice, FH-4419, is paid.",
            "Your invoice, F, as in frank. H, as in house. Four, four, one, nine, is paid.",
        ),
        ("(FH-4419)", "(F, as in frank. H, as in house. Four, four, one, nine.)"),
    ],
)
def test_ids(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1", "one dollar"),
        ("$1.00", "one dollar"),
        ("$1.01", "one dollar and one cent"),
        ("$0.50", "fifty cents"),
        ("$0.01", "one cent"),
        ("$0", "zero dollars"),
        ("$0.00", "zero dollars"),
        ("$50", "fifty dollars"),
        ("$100.00", "one hundred dollars"),
        ("$14.99", "fourteen dollars and ninety nine cents"),
        ("$1,234.56", "one thousand two hundred thirty four dollars and fifty six cents"),
        ("$1,000,000", "one million dollars"),
        ("$5.5", "five dollars and fifty cents"),
        ("-$5.00", "minus five dollars"),
        ("$19.99/mo", "nineteen dollars and ninety nine cents a month"),
        ("$150/year", "one hundred fifty dollars a year"),
        ("It was $49.99.", "It was forty nine dollars and ninety nine cents."),
        ("$49.99, then $12.", "forty nine dollars and ninety nine cents, then twelve dollars."),
    ],
)
def test_money(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("imap.fernhill.example", "eye map dot fernhill dot example"),
        ("smtp.fernhill.example", "S M T P dot fernhill dot example"),
        ("www.fernhill.example", "W W W dot fernhill dot example"),
        ("billing.fernhill.example.", "billing dot fernhill dot example."),
        ("example.com", "example dot com"),
        ("my-site.example", "my dash site dot example"),
        ("site24.example", "site two, four dot example"),
        ("whitfieldlandscaping.example", "whitfield landscaping dot example"),
        ("https://fernhill.example", "fernhill dot example"),
        ("https://fernhill.example/", "fernhill dot example"),
        (
            "https://www.fernhill.example/help/billing?id=1",
            "W W W dot fernhill dot example slash help slash billing",
        ),
        ("fernhill.example/help", "fernhill dot example slash help"),
    ],
)
def test_hostnames_and_urls(text, expected):
    assert spoken(text) == expected


def test_a_typo_is_not_a_hostname():
    assert spoken("All done.Thanks") == "All done.Thanks"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dana@whitfieldlandscaping.example", "dana at whitfield landscaping dot example"),
        ("priya.nair@mailbox.example", "priya dot nair at mailbox dot example"),
        ("first_last@a-b.example", "first underscore last at a dash b dot example"),
        ("sales+news@x.com", "sales plus news at x dot com"),
        ("bob2@x.com", "bob two at x dot com"),
        ("Write to info@x.com.", "Write to info at x dot com."),
    ],
)
def test_email_addresses(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(503) 555-0101", "five, zero, three, five, five, five, zero, one, zero, one"),
        ("503-555-0101", "five, zero, three, five, five, five, zero, one, zero, one"),
        ("503.555.0101", "five, zero, three, five, five, five, zero, one, zero, one"),
        ("503 555 0101", "five, zero, three, five, five, five, zero, one, zero, one"),
        ("+1 503 555 0101", "one, five, zero, three, five, five, five, zero, one, zero, one"),
        ("1-800-555-0199", "one, eight, zero, zero, five, five, five, zero, one, nine, nine"),
        ("555-0199", "five, five, five, zero, one, nine, nine"),
        ("5035550101", "five, zero, three, five, five, five, zero, one, zero, one"),
        (
            "Call 503-555-0101 today.",
            "Call five, zero, three, five, five, five, zero, one, zero, one today.",
        ),
    ],
)
def test_phone_numbers(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-06", "November sixth, twenty twenty six"),
        ("2027-01-01", "January first, twenty twenty seven"),
        ("11/06/2026", "November sixth, twenty twenty six"),
        ("1/2/27", "January second, twenty twenty seven"),
        ("Nov 6, 2026", "November sixth, twenty twenty six"),
        ("Nov. 6", "November sixth"),
        ("November 22nd", "November twenty second"),
        ("October 30 2026", "October thirtieth, twenty twenty six"),
        ("Sept 3", "September third"),
    ],
)
def test_dates(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize("text", ["2026-13-45", "Feb 30", "2/30/2026"])
def test_impossible_dates_are_read_as_digits_not_as_a_wrong_date(text):
    result = spoken(text)
    assert not any(month in result for month in ("February", "January", "first", "thirtieth"))
    assert "," in result  # read out one digit at a time


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10 GB", "ten gigabytes"),
        ("10GB", "ten gigabytes"),
        ("1 GB", "one gigabyte"),
        ("1.5 GB", "one point five gigabytes"),
        ("500 MB", "five hundred megabytes"),
        ("2 TB", "two terabytes"),
        ("25gb", "twenty five gigabytes"),
        ("1,024 MB", "one thousand twenty four megabytes"),
        ("How many GB is that?", "How many gigabytes is that?"),
    ],
)
def test_sizes(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DNS", "D N S"),
        ("IMAP", "eye map"),
        ("SMTP", "S M T P"),
        ("SSL", "S S L"),
        ("MX", "M X"),
        ("TTL", "T T L"),
        ("HTTPS", "H T T P S"),
        ("STARTTLS", "start T L S"),
        ("POP3", "pop three"),
        ("CNAME", "C name"),
        ("dns", "D N S"),
        ("Your SSL certificate", "Your S S L certificate"),
        ("the MX record.", "the M X record."),
        ("Mr. Nguyen", "Mr. win"),
    ],
)
def test_lexicon_terms(text, expected):
    assert spoken(text) == expected


def test_terms_do_not_match_inside_words():
    assert spoken("mixed axis") == "mixed axis"
    assert spoken("identity") == "identity"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("port 993", "port nine, nine, three"),
        ("Port 587", "Port five, eight, seven"),
        ("ports 993 and 587", "ports nine, nine, three and five, eight, seven"),
        ("Your PIN is 4821.", "Your pin is four, eight, two, one."),
        ("zip code 97214", "zip code nine, seven, two, one, four"),
        ("billing zip is 97214", "billing zip is nine, seven, two, one, four"),
        ("card ending in 4242", "card ending in four, two, four, two"),
        ("invoice 4419", "invoice four, four, one, nine"),
        ("ticket number 2203", "ticket number two, two, zero, three"),
        ("97214", "nine, seven, two, one, four"),
        ("0421", "zero, four, two, one"),
    ],
)
def test_numbers_read_digit_by_digit(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10 minutes", "ten minutes"),
        ("24 hours", "twenty four hours"),
        ("30 days", "thirty days"),
        ("0", "zero"),
        ("TTL 3600", "T T L three thousand six hundred"),
        ("priority 10", "priority ten"),
        ("1.5", "one point five"),
        ("2.50", "two point five zero"),
        ("1,234", "one thousand two hundred thirty four"),
        ("in 2026", "in twenty twenty six"),
        ("20%", "twenty percent"),
        ("2.5%", "two point five percent"),
        ("the 5th", "the fifth"),
        ("the 22nd", "the twenty second"),
    ],
)
def test_quantities(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8 AM", "eight A M"),
        ("6pm", "six P M"),
        ("5:30 PM", "five thirty P M"),
        ("12:05 pm", "twelve oh five P M"),
        ("8:00 AM", "eight A M"),
        ("5 p.m.", "five P M"),
        ("9:30", "nine thirty"),
        ("5:00", "five o'clock"),
        ("24/7", "twenty four seven"),
    ],
)
def test_times(text, expected):
    assert spoken(text) == expected


def test_ip_address_is_not_a_hostname_or_number():
    assert spoken("203.0.113.11") == "two, zero, three dot zero dot one, one, three dot one, one"


def test_one_rule_never_rewrites_another_rules_output():
    # The hostname rule says "S M T P"; a later term pass must not see its digits or words.
    assert spoken("Use smtp.fernhill.example port 587 with STARTTLS.") == (
        "Use S M T P dot fernhill dot example port five, eight, seven with start T L S."
    )


def test_a_full_support_sentence():
    assert spoken(
        "Set the MX record for nguyenauto.example to mail.fernhill.example, TTL 3600."
    ) == (
        "Set the M X record for win auto dot example to mail dot fernhill dot example, "
        "T T L three thousand six hundred."
    )


def test_plain_speech_passes_through():
    text = "Sure, let me look that up for you."
    assert normalize(text) == normalize(text)
    assert spoken(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Important:** read this", "Important: read this"),
        ("# Heading\nBody", "Heading\nBody"),
        ("- one\n- two", "one\ntwo"),
        ("1. one\n2. two", "one\ntwo"),
        ("Use `port 993`", "Use port 993"),
        ("See [the guide](https://x.example/a)", "See the guide"),
        ("Done \U0001f600", "Done"),
    ],
)
def test_markup_is_removed(text, expected):
    assert clean_markup(text) == expected


def test_markup_never_reaches_tts():
    assert (
        spoken("**Your** invoice:\n- FH-4419")
        == "Your invoice: F, as in frank. H, as in house. Four, four, one, nine."
    )


def test_dashes_become_pauses():
    assert spoken("One moment — checking now") == "One moment, checking now"


def test_ampersand():
    assert spoken("Bell & Sons") == "Bell and Sons"


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (20, "twenty"),
        (42, "forty two"),
        (100, "one hundred"),
        (101, "one hundred one"),
        (1000, "one thousand"),
        (1234, "one thousand two hundred thirty four"),
        (1000000, "one million"),
        (-3, "minus three"),
    ],
)
def test_number_to_words(number, expected):
    assert number_to_words(number) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (5, "fifth"),
        (8, "eighth"),
        (9, "ninth"),
        (12, "twelfth"),
        (20, "twentieth"),
        (21, "twenty first"),
        (30, "thirtieth"),
        (31, "thirty first"),
    ],
)
def test_ordinals(number, expected):
    assert ordinal_to_words(number) == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2026, "twenty twenty six"),
        (2000, "two thousand"),
        (2005, "two thousand five"),
        (2010, "twenty ten"),
        (1999, "nineteen ninety nine"),
        (1900, "nineteen hundred"),
        (1905, "nineteen oh five"),
    ],
)
def test_years(year, expected):
    assert year_to_words(year) == expected


def test_lexicon_loads_from_seeds():
    lexicon = load_lexicon()
    assert isinstance(lexicon, Lexicon)
    assert lexicon.terms["IMAP"] == "eye map"
    assert lexicon.units["GB"] == ("gigabyte", "gigabytes")
    assert lexicon.spoken_term("smtp") == "S M T P"


def test_a_custom_lexicon_is_honored():
    lexicon = Lexicon(terms={"DNS": "domain name system"}, units={}, tlds=("example",))
    assert normalize("DNS", lexicon).spoken == "domain name system"


@pytest.mark.parametrize(
    ("text", "sentences", "tail"),
    [
        ("Hello there. How are you? Good", ["Hello there.", "How are you?"], "Good"),
        ("All done!", [], "All done!"),
        ("All done! ", ["All done!"], ""),
        ("It costs $49.", [], "It costs $49."),
        ("It costs $49.99 a month. Okay", ["It costs $49.99 a month."], "Okay"),
        (
            "Use mail.fernhill.example for that. Okay",
            ["Use mail.fernhill.example for that."],
            "Okay",
        ),
        ("Dr. Smith called. Bye", ["Dr. Smith called."], "Bye"),
        ("Use e.g. this one. Bye", ["Use e.g. this one."], "Bye"),
        ("J. Smith called. Bye", ["J. Smith called."], "Bye"),
        ("1. Go there. Bye", ["1. Go there."], "Bye"),
        ("First line\nSecond line\n", ["First line", "Second line"], ""),
        ('He said "wait." Then left. ', ['He said "wait."', "Then left."], ""),
        ("Really?! Yes. ", ["Really?!", "Yes."], ""),
        ("", [], ""),
    ],
)
def test_split_sentences(text, sentences, tail):
    assert split_sentences(text) == (sentences, tail)


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("It expires on Nov. 6, 2027. Thanks. ", "It expires on Nov. 6, 2027."),
        ("We open Sept. 3 now. Thanks. ", "We open Sept. 3 now."),
        ("Pay by Dec. 1. Thanks. ", "Pay by Dec. 1."),
        ("Call Mon. through Fri. please. Thanks. ", "Call Mon. through Fri. please."),
        ("Call ext. 123 now. Thanks. ", "Call ext. 123 now."),
        ("We open at 8 a.m. Pacific today. Thanks. ", "We open at 8 a.m. Pacific today."),
        ("It costs approx. $5 a month. Thanks. ", "It costs approx. $5 a month."),
        ("Jensen Plumbing Inc. is on file. Thanks. ", "Jensen Plumbing Inc. is on file."),
        ("See the U.S. office. Thanks. ", "See the U.S. office."),
        ("Ask for Sec. 2 now. Thanks. ", "Ask for Sec. 2 now."),
    ],
)
def test_abbreviations_do_not_end_a_sentence(text, first):
    assert split_sentences(text)[0] == [first, "Thanks."]


@pytest.mark.parametrize(
    "ending", ["The sun. ", "He sat. ", "We were in the mar. ", "Tell the fan. "]
)
def test_lowercase_words_that_look_like_abbreviations_still_end_a_sentence(ending):
    assert split_sentences(f"{ending}Thanks. ")[0] == [ending.strip(), "Thanks."]


def test_splitting_is_stable_as_text_streams_in():
    text = "Your total is $49.99. It renews on Nov. 6. Anything else?"
    seen = []
    for end in range(len(text)):
        finished, _ = split_sentences(text[:end])
        assert finished[: len(seen)] == seen
        seen = finished
