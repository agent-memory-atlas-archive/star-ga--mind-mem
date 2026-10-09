# ruff: noqa: E501  (tables of natural-language test sentences)
"""Every input any verifier used to break fact_slots, as regression tables.

Merge rule (slots_differ): a merge may be proposed only when every difference
is plain wording. Contradiction rule (changed_fact): a correction differs only
in when/zone values, plus correction words on the new side.
"""

from __future__ import annotations

import time

import pytest

from mind_mem.fact_slots import changed_fact, slots_differ
from mind_mem.granularity_align import find_merge_candidates

# (a, b): these must NEVER be proposed as a merge.
NEVER_MERGE = [
    ("Ilona call Tue 10am Arizona time.", "Contractor call Tue 10am Pacific time."),
    ("Weekly planning call with Ilona on Tuesday at 10am PT.", "Weekly planning call with Ilona on Tuesday at 10am ET."),
    ("Call with Ilona on Tuesday at 10am in Pacific.", "Call with Ilona on Tuesday at 10am in Eastern."),
    ("Call with José on Tuesday at 10am.", "Call with Joséphine on Tuesday at 10am."),
    ("Call with José on Tuesday at 10am.", "Call with Joséphine on Tuesday at 10am."),  # NFD
    ("Call with Ángel on Tuesday at 10am.", "Call with Óscar on Tuesday at 10am."),
    ("Weekly planning call with 李明 on Tuesday at 10am.", "Weekly planning call with 王芳 on Tuesday at 10am."),
    ("Lunch with Jan on Friday at noon.", "Lunch with May on Friday at noon."),
    ("Rent payment to Ilona is due on the 5th of each month.", "Rent payment to Ilona is due on the 15th of each month."),
    ("Budget review at 10am in 2025.", "Budget review at 10am in 2026."),
    ("Standup today at 10am.", "Standup tomorrow at 10am."),
    ("Standup at 10 in the morning.", "Standup at 10 in the evening."),
    ("Standup at 10h.", "Standup at 14h."),
    ("Standup at 1000 hours.", "Standup at 1400 hours."),
    ("Standup at 9.30.", "Standup at 9.45."),
    ("Standup at 0930.", "Standup at 0945."),
    ("Standup at half past nine.", "Standup at quarter to ten."),
    ("Ilona: Tuesday. 10am: Marcus.", "Ilona: 10am. Tuesday: Marcus."),
    ("Ilona: Tuesday\n10am: Marcus", "Ilona: 10am\nTuesday: Marcus"),
    ("Ilona Tuesday, PT Marcus", "Ilona PT, Tuesday Marcus"),
    ("Ilona PT, 10am Marcus ET", "Ilona 10am, PT Marcus ET"),
    ("Budget item 1. 5 engineers approved", "Budget item 1.5 engineers approved"),
    ("Decision 1.\n5 engineers join Ilona", "Decision 1.5 engineers join Ilona"),
    ("Rooms 1, 5 booked", "Rooms 1,5 booked"),
    ("Marcus ordered 10 May shipments", "Marcus ordered May 10 shipments"),
    ("Ship 5 May orders", "Ship May 5 orders"),
    ("Office hours Tue-Fri.", "Office hours Tue, Fri."),
    ("Ilona calls Marcus every Monday.", "Marcus calls Ilona every Monday."),
    ("Azariel must never share the investor deck outside the board.", "Azariel must share the investor deck outside the board."),
    ("Azariel must share the deck.", "Azariel may share the deck."),
    ("Meet at 10:30.", "Meet at 10:45."),
    ("Meet at 10am UTC+2.", "Meet at 10am UTC+3."),
]

# (a, b): plain-wording duplicates that SHOULD still be merge candidates.
SHOULD_MERGE = [
    (
        "Weekly planning call with Ilona on Tuesday at 10am Arizona time.",
        "Weekly planning call with Ilona, Tuesday 10am Arizona time, recurring.",
    ),
    ("fixed-point arithmetic ensures determinism across substrates", "use fixed-point arithmetic for cross-substrate determinism"),
    ("the release workflow publishes wheels to PyPI on tag push", "release workflow publishes the wheels to PyPI on a tag push"),
    ("Call with Ilona on Tue at 10 am.", "Call with Ilona on Tuesday at 10am."),
    # same fact, phrase moved; "sat" is the verb, never Saturday
    ("Ilona sat at 10am for the exam.", "Ilona sat for the exam at 10am."),
]

# (new, old): corrections that MUST be flagged.
CONTRADICTS = [
    (
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
        "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    (
        "Weekly planning call with Ilona on Tuesday at 11am Arizona time.",
        "Weekly planning call with Ilona on Tuesday at 10am Arizona time.",
    ),
    ("Ilona's planning call is on Thursdays at 10am.", "Ilona's planning call is on Tuesdays at 10am."),
    (
        "Azariel's weekly planning call with Ilona moved to Thursday at 10am Arizona time; it is no longer on Tuesday.",
        "Azariel's weekly planning call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    ("The planning call is now at 11am.", "The planning call is at 10am."),
    ("Board review on 2026-10-22.", "Board review on 2026-10-15."),
    ("Standup at 10:30.", "Standup at 9:30."),
]

# (new, old): different statements that must NOT be flagged as contradictions.
NOT_CONTRADICTS = [
    (
        "Azariel's weekly planning call with the engineering contractor is on Tuesday at 10am Pacific time.",
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
    ),
    (
        "Weekly planning call with Ilona: Thursday, 10am Arizona time.",
        "Azariel's weekly planning call with Ilona is on Thursday at 10am Arizona time.",
    ),
    ("Ilona's dentist visit is on Friday, the day after the planning call.", "Ilona's planning call is on Thursday."),
    ("Dentist visit on Friday, after the planning call.", "Planning call on Tuesday."),
    ("The daily lunch is at noon.", "The daily standup is at 9am."),
    ("Azariel's weekly dentist visit is on Friday.", "Azariel's weekly call is on Tuesday."),
    ("Ilona's return flight on Thursday.", "Ilona's flight on Tuesday."),
    ("Ilona's final exam on May 12.", "Ilona's exam on May 5."),
    ("Planning call prep at 9am.", "Planning call at 10am."),
    ("Call with Ilona's lawyer on Thursday.", "Call with Ilona on Tuesday."),
    ("Dentist appointment for son on Thursday.", "Dentist appointment on Tuesday."),
    ("Lunch with Marcus on Friday.", "Lunch with Ilona on Thursday."),
]

CRASH_INPUTS = ["Office on ²nd floor", "Room ³rd", "Meet ¹st May", "x ①a", "٣ apples", "\x00\x01", "", "   ", "..."]


@pytest.mark.parametrize(("a", "b"), NEVER_MERGE)
def test_never_merge(a, b):
    assert slots_differ(a, b)
    assert slots_differ(b, a)
    blocks = [{"_id": "A", "content": a}, {"_id": "B", "content": b}]
    assert find_merge_candidates(blocks, min_similarity=0.0) == []


@pytest.mark.parametrize(("a", "b"), SHOULD_MERGE)
def test_wording_only_duplicates_still_merge(a, b):
    assert not slots_differ(a, b)


@pytest.mark.parametrize(("new", "old"), CONTRADICTS)
def test_changed_fact_is_flagged(new, old):
    assert changed_fact(new, old)


@pytest.mark.parametrize(("new", "old"), NOT_CONTRADICTS)
def test_different_statement_is_not_a_contradiction(new, old):
    assert not changed_fact(new, old)


@pytest.mark.parametrize("text", CRASH_INPUTS)
def test_never_raises(text):
    slots_differ(text, "Office on 2nd floor")
    changed_fact(text, "Office on 2nd floor")
    find_merge_candidates([{"_id": "A", "content": text}, {"_id": "B", "content": text}], min_similarity=0.0)


@pytest.mark.parametrize(
    "text",
    [
        " " * 300_000 + "x",
        "1" * 300_000,
        ":" * 300_000,
        "1:" * 150_000,
        "(" * 150_000 + ")" * 150_000,
        "Tue 10am PT, " * 25_000,
    ],
)
def test_linear_time(text):
    start = time.perf_counter()
    slots_differ(text, text + " x")
    changed_fact(text + " moved", text)
    assert time.perf_counter() - start < 3.0
