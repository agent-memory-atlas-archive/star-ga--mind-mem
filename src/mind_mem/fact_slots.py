# Copyright 2026 STARGA, Inc.
"""Fail-closed fact comparison for memory merges and contradictions.

Two decisions read this module:

* ``slots_differ(a, b)`` — may two memories be MERGED? Only when they are the
  same statement after a fixed, conservative normalisation. A wrongly
  proposed merge loses a memory; a missed merge costs nothing (merges are
  review-gated proposals). So the rule is equality, never "no difference
  detected": every token is compared, in order, punctuation included.
* ``changed_fact(new, old)`` — is ``new`` a correction of ``old``? Only when
  the two statements align token for token except where a day, date, clock
  time or time zone changed (plus correction words such as "moved", "no
  longer" on the new side). A different person, event or any other word is a
  different statement, not a contradiction.

Normalisation is deliberately small: Unicode NFC, case folding, whitespace
collapse, an exact table of unambiguous abbreviations, and joining a number
directly followed by am/pm. No reordering, no sorting, no joining across
punctuation, no fuzzy matching. Only ASCII 0-9 count as digits.

Deterministic, linear time, no I/O.
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass

__all__ = ["FactSlots", "changed_fact", "fact_slots", "normalized_tokens", "slots_differ"]

_ASCII_DIGITS = frozenset("0123456789")

# Exact, unambiguous spellings only. Words that are also ordinary English
# (mon, wed, sat, sun, mar, may, jan as a name...) are deliberately absent:
# they stay themselves, so they can only ever match their own spelling.
_EQUIV = {
    "tue": "tuesday",
    "tues": "tuesday",
    "tuesdays": "tuesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "thursdays": "thursday",
    "fri": "friday",
    "fridays": "friday",
    "mondays": "monday",
    "wednesdays": "wednesday",
    "saturdays": "saturday",
    "sundays": "sunday",
    "feb": "february",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
_ARTICLES = frozenset({"a", "an", "the"})

_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)
_ZONES = frozenset(
    {
        "utc",
        "gmt",
        "pt",
        "pst",
        "pdt",
        "et",
        "est",
        "edt",
        "ct",
        "cst",
        "cdt",
        "mt",
        "mst",
        "mdt",
        "pacific",
        "eastern",
        "central",
        "mountain",
        "arizona",
        "hawaii",
        "alaska",
        "cet",
        "cest",
        "bst",
        "jst",
        "aest",
        "aedt",
    }
)
_RELATIVE_DAYS = frozenset({"today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "noon", "midnight"})
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion half quarter o'clock oclock first second third fourth fifth sixth seventh eighth ninth tenth".split()
)
# Words a correction may add on the NEW side without changing the subject.
_CORRECTION = frozenset(
    {
        "moved",
        "move",
        "now",
        "no",
        "longer",
        "instead",
        "rescheduled",
        "changed",
        "to",
        "from",
        "not",
        "it",
        "is",
        "was",
        "has",
        "been",
        "will",
        "be",
        "on",
        "at",
        "of",
        "and",
        "but",
    }
)
_PUNCT_IGNORED_FOR_CHANGE = frozenset(".,;:!?()[]\"'")


def _is_ascii_number(tok: str) -> bool:
    return bool(tok) and all(c in _ASCII_DIGITS for c in tok)


def _raw_tokens(text: str) -> list[tuple[str, str]]:
    """Split into (folded, original) tokens: word runs and single punctuation characters.

    A word run is Unicode letters/marks plus ASCII digits. A '.', ',' or ':'
    with an ASCII digit directly on both sides stays inside the run, so
    "1.5", "1,000" and "10:30" are one token while "1. 5" is three.
    """
    text = unicodedata.normalize("NFC", text)
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        is_word = ch in _ASCII_DIGITS or unicodedata.category(ch)[0] in ("L", "M")
        if not is_word and ch in ".,:" and cur and cur[-1] in _ASCII_DIGITS and i + 1 < n and text[i + 1] in _ASCII_DIGITS:
            is_word = True
        if is_word:
            cur.append(ch)
            continue
        if cur:
            word = "".join(cur)
            out.append((word.casefold(), word))
            cur = []
        if not ch.isspace():
            out.append((ch.casefold(), ch))  # every other non-space char is its own token
    if cur:
        word = "".join(cur)
        out.append((word.casefold(), word))
    return out


def _normalized(text: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    for folded, orig in _raw_tokens(text):
        folded = _EQUIV.get(folded, folded)
        if folded in _ARTICLES:
            continue
        # "10 am" == "10am": join a number DIRECTLY followed by am/pm (no punctuation between).
        if folded in ("am", "pm") and toks and _is_ascii_number(toks[-1][0]):
            prev_f, prev_o = toks[-1]
            toks[-1] = (prev_f + folded, prev_o + orig)
            continue
        toks.append((folded, orig))
    return toks


def normalized_tokens(text: str) -> tuple[str, ...]:
    """The token sequence both decisions compare (order and punctuation preserved)."""
    return tuple(f for f, _ in _normalized(text))


def _is_value(tok: str) -> bool:
    """A when/zone/quantity value: weekday, month, relative day, zone, or anything with a digit."""
    if tok in _WEEKDAYS or tok in _MONTHS or tok in _ZONES or tok in _RELATIVE_DAYS or tok in _NUMBER_WORDS:
        return True
    return any(c in _ASCII_DIGITS for c in tok)


# A change in any of these changes what a statement asserts, so it is never wording.
_POLARITY = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nor",
        "without",
        "must",
        "mustn",
        "should",
        "shouldn",
        "shall",
        "may",
        "might",
        "can",
        "cannot",
        "could",
        "couldn",
        "will",
        "won",
        "would",
        "wouldn",
        "don",
        "doesn",
        "didn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "always",
        "only",
        "except",
        "unless",
    }
)


def _wording_only(folded: str, orig: str) -> bool:
    """True when a differing token is plain wording, not a fact-bearing token.

    Fact-bearing (blocks a merge): any when/zone/quantity value, any token with
    a non-ASCII letter (names in other scripts), any token capitalised in the
    original text (names; also sentence-initial words — fail-closed), and any
    polarity or modal word.
    """
    if not folded.strip():
        return True
    if all(not c.isalnum() for c in folded):
        return True  # punctuation
    if _is_value(folded) or folded in _POLARITY:
        return False
    if not orig.isascii() or not orig.isalpha():
        return False
    return orig == orig.lower()


@dataclass(frozen=True)
class FactSlots:
    """The when/zone values named in a text, plus its full normalised token sequence."""

    tokens: tuple[str, ...]
    values: tuple[str, ...]


def fact_slots(text: str) -> FactSlots:
    toks = normalized_tokens(text)
    return FactSlots(tokens=toks, values=tuple(t for t in toks if _is_value(t)))


#: Above this many differing tokens (after trimming the common prefix and
#: suffix) the texts are not near-duplicates; both decisions fail closed.
_MAX_DIFF_WINDOW = 2_000


def _opcodes(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]] | None:
    """Alignment opcodes in near-linear time, or None when the differing window is too large.

    The common prefix and suffix are matched in O(n); only the differing
    middle goes through SequenceMatcher, which is quadratic in the worst case.
    """
    n, m = len(a), len(b)
    p = 0
    while p < n and p < m and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and s < m - p and a[n - 1 - s] == b[m - 1 - s]:
        s += 1
    mid_a, mid_b = a[p : n - s], b[p : m - s]
    if len(mid_a) > _MAX_DIFF_WINDOW or len(mid_b) > _MAX_DIFF_WINDOW:
        return None
    ops: list[tuple[str, int, int, int, int]] = []
    if p:
        ops.append(("equal", 0, p, 0, p))
    if mid_a or mid_b:
        sm = difflib.SequenceMatcher(a=mid_a, b=mid_b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            ops.append((tag, i1 + p, i2 + p, j1 + p, j2 + p))
    if s:
        ops.append(("equal", n - s, n, m - s, m))
    return ops


def slots_differ(a: str, b: str) -> bool:
    """True unless every difference between the two texts is plain wording.

    The texts are aligned in order (so swapped people or values show up as
    changed spans). Any changed span that contains a fact-bearing token — see
    :func:`_wording_only` — means the two memories may say different things,
    and a merge would lose one of them. Missing a real duplicate is
    acceptable; proposing a false merge is not.
    """
    ta, tb = _normalized(a), _normalized(b)
    fa, fb = [f for f, _ in ta], [f for f, _ in tb]
    if fa == fb:
        return False
    ops = _opcodes(fa, fb)
    if ops is None:
        return True
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        for seq, lo, hi in ((ta, i1, i2), (tb, j1, j2)):
            for k in range(lo, hi):
                f, o = seq[k]
                if not _wording_only(f, o) or _joins_values(seq, k):
                    return True
    return False


def _joins_values(seq: list[tuple[str, str]], k: int) -> bool:
    """A punctuation token with a value on both sides ("Tue-Fri", "Tue, Fri", "5/7") is fact-bearing."""
    f = seq[k][0]
    if any(c.isalnum() for c in f):
        return False
    return 0 < k < len(seq) - 1 and _is_value(seq[k - 1][0]) and _is_value(seq[k + 1][0])


def _content(tokens: tuple[str, ...]) -> list[str]:
    return [t for t in tokens if t not in _PUNCT_IGNORED_FOR_CHANGE]


def changed_fact(new: str, old: str) -> bool:
    """True when ``new`` restates ``old`` with a different day/date/time/zone and nothing else.

    Every differing span must consist of when/zone values or correction and
    function words ("moved", "no longer", "is", "on"...), on either side. At
    least 60% of the old statement's tokens must match in order (same
    subject), and the two statements' value sets must differ.
    """
    o = _content(normalized_tokens(old))
    n = _content(normalized_tokens(new))
    if not o or not n:
        return False
    ops = _opcodes(o, n)
    if ops is None:
        return False
    equal = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            equal += i2 - i1
            continue
        for t in o[i1:i2] + n[j1:j2]:
            if not (_is_value(t) or t in _CORRECTION):
                return False  # a non-value, non-correction word changed: different statement
    if equal * 10 < len(o) * 6:
        return False
    old_vals = {t for t in o if _is_value(t)}
    new_vals = {t for t in n if _is_value(t)}
    if not old_vals or not new_vals:
        return False
    return old_vals != new_vals
