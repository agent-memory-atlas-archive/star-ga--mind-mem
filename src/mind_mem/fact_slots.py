# Copyright 2026 STARGA, Inc.
"""Fail-closed fact comparison for memory merges and contradictions.

Two decisions read this module, and both are designed so that words alone
can decide them SAFELY:

* ``slots_differ(a, b)`` — may two memories be MERGED? Only when they are the
  same statement after a meaning-preserving normalisation: Unicode NFC, case
  folding, whitespace, a trailing full stop, an exact table of unambiguous
  abbreviations ("tue" = "tuesday") and canonical clock times ("10am" =
  "10:00am" = "10 a.m."). Every other word, number, symbol and emoji must be
  identical, in order. A wrongly proposed merge loses a memory; deciding that
  two DIFFERENTLY worded sentences mean the same thing is a semantic judgement
  that no word list can make safely, so it is never made here.
* ``changed_fact(new, old)`` — is ``new`` a correction of ``old``? Strip the
  when/zone values (days, months, dates, clock times, zones, relative days)
  and a fixed set of correction and function words from both statements. If
  what remains (people, event, identifiers, quantities, every other word) is
  identical, and the value sets differ or the negation flips, it is a changed
  fact. Plain numbers are identifiers or quantities, never when-values, so
  "Bug 1234" vs "Bug 5678" is not a contradiction.

Both decisions are a single linear pass: no alignment, no fuzzy matching.
Only ASCII 0-9 count as digits. Deterministic, no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["FactSlots", "canonical_tokens", "changed_fact", "fact_slots", "slots_differ"]

_ASCII_DIGITS = frozenset("0123456789")
# Kept inside a token when an ASCII digit is directly on both sides:
# "1.5", "1,000", "10:30", "2026-10-08", "10/8".
_DIGIT_JOINERS = frozenset(".,:-/")

# Exact, unambiguous spellings only (applied to merges and contradictions alike).
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
# Ambiguous with ordinary words: treated as calendar values ONLY by changed_fact
# (where everything else must already be identical); never rewritten for merges.
_AMBIGUOUS_CALENDAR = {"mon": "monday", "wed": "wednesday", "sat": "saturday", "sun": "sunday", "jan": "january", "mar": "march"}

_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
)
_ZONES = frozenset(
    "utc gmt pt pst pdt et est edt ct cst cdt mt mst mdt hst akst akdt ist cet cest bst jst aest aedt "
    "pacific eastern central mountain arizona hawaii alaska".split()
)
_RELATIVE = frozenset(
    "today tomorrow yesterday tonight morning afternoon evening noon midnight weekend weekday next this last week month year".split()
)
# Words a correction may add or drop without changing what the statement is about.
_CORRECTION = frozenset(
    "moved move now no longer instead rescheduled reschedule changed change switched postponed delayed pushed back got get "
    "to from for it is was are were has have had been be on at of and but correction actually update updated "
    "again in by until starting as the a an".split()
)
_NEGATION = frozenset({"not", "never"})
# "isn't" == "is not", "can't"/"cannot" == "can not": meaning-preserving, so merges use it too.
_CONTRACTION_BASE = {
    "isn": "is",
    "aren": "are",
    "wasn": "was",
    "weren": "were",
    "don": "do",
    "doesn": "does",
    "didn": "did",
    "won": "will",
    "can": "can",
    "ca": "can",
    "couldn": "could",
    "shouldn": "should",
    "wouldn": "would",
    "hasn": "has",
    "haven": "have",
    "hadn": "had",
    "mustn": "must",
    "needn": "need",
}

_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_SLASH_DATE = re.compile(r"^\d{1,4}/\d{1,2}(/\d{1,4})?$")
_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$")
_CANON_CLOCK = re.compile(r"^t\d{2}:\d{2}$")
_ORDINAL_DAY = re.compile(r"^\d{1,2}(st|nd|rd|th)$")


def _is_digits(tok: str) -> bool:
    return bool(tok) and all(c in _ASCII_DIGITS for c in tok)


def _raw_tokens(text: str) -> list[str]:
    """Case-folded word runs (letters, marks, ASCII digits) and single non-space characters."""
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    cur: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        is_word = ch in _ASCII_DIGITS or unicodedata.category(ch)[0] in ("L", "M")
        if not is_word and ch in _DIGIT_JOINERS and cur and cur[-1] in _ASCII_DIGITS and i + 1 < n and text[i + 1] in _ASCII_DIGITS:
            is_word = True
        if is_word:
            cur.append(ch)
            continue
        if cur:
            out.append("".join(cur).casefold())
            cur = []
        if not ch.isspace():
            out.append(ch.casefold())
    if cur:
        out.append("".join(cur).casefold())
    return out


def _canon_clock(hour: str, minute: str | None, meridiem: str | None) -> str | None:
    h = int(hour)
    m = int(minute) if minute else 0
    if meridiem:
        if not 1 <= h <= 12 or m > 59:
            return None
        h = h % 12 + (12 if meridiem == "pm" else 0)
    elif minute is None or h > 23 or m > 59:
        return None  # a bare number is not a clock time
    return f"t{h:02d}:{m:02d}"


_MERIDIEM_FORMS = (
    (("a", ".", "m", "."), "am"),
    (("p", ".", "m", "."), "pm"),
    (("a", ".", "m"), "am"),
    (("p", ".", "m"), "pm"),
    (("am",), "am"),
    (("pm",), "pm"),
)


def canonical_tokens(text: str) -> tuple[str, ...]:
    """The meaning-preserving normal form both decisions compare."""
    raw = _raw_tokens(text)
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        tok = _EQUIV.get(raw[i], raw[i])
        if tok in _CONTRACTION_BASE and i + 2 < n and raw[i + 1] in ("'", "\u2019") and raw[i + 2] == "t":
            out.extend((_CONTRACTION_BASE[tok], "not"))
            i += 3
            continue
        if tok == "cannot":
            out.extend(("can", "not"))
            i += 1
            continue
        m = _CLOCK.match(tok)
        if m:
            mer = m.group(3)
            j = i + 1
            if mer is None:
                for form, value in _MERIDIEM_FORMS:
                    if tuple(raw[j : j + len(form)]) == form:
                        mer, j = value, j + len(form)
                        break
            clock = _canon_clock(m.group(1), m.group(2), mer)
            if clock is not None:
                out.append(clock)
                i = j
                continue
        out.append(tok)
        i += 1
    while out and out[-1] in (".", "。"):
        out.pop()  # a trailing full stop carries no meaning
    return tuple(out)


@dataclass(frozen=True)
class FactSlots:
    """A statement split into its when/zone values and everything else (the core)."""

    core: tuple[str, ...]
    values: frozenset[str]
    negations: int


def fact_slots(text: str) -> FactSlots:
    toks = [_AMBIGUOUS_CALENDAR.get(t, t) for t in canonical_tokens(text)]
    values: set[str] = set()
    core: list[str] = []
    negations = 0
    n = len(toks)
    for i, t in enumerate(toks):
        if _CANON_CLOCK.match(t) or t in _WEEKDAYS or t in _MONTHS or t in _ZONES or t in _RELATIVE:
            values.add(t)
        elif _ISO_DATE.match(t) or _SLASH_DATE.match(t):
            values.add("d" + t)
        elif (_is_digits(t) or _ORDINAL_DAY.match(t)) and ((i > 0 and toks[i - 1] in _MONTHS) or (i + 1 < n and toks[i + 1] in _MONTHS)):
            values.add("n" + t.rstrip("stndrh"))  # a day/year number next to a month is part of the date
        elif _ORDINAL_DAY.match(t) and 0 < i < n - 1 and toks[i - 1] == "the" and toks[i + 1] == "of":
            values.add("n" + t.rstrip("stndrh"))  # "the 5th of each month"
        elif t in _NEGATION:
            negations += 1
        elif t in _CORRECTION:
            continue
        elif len(t) == 1 and unicodedata.category(t)[0] == "P":
            continue  # punctuation only; symbols ($, €, +, <, emoji) stay in the core
        else:
            core.append(t)
    return FactSlots(core=tuple(core), values=frozenset(values), negations=negations)


def slots_differ(a: str, b: str) -> bool:
    """True unless the two texts are the same statement after the meaning-preserving normalisation."""
    return canonical_tokens(a) != canonical_tokens(b)


def changed_fact(new: str, old: str) -> bool:
    """True when ``new`` corrects ``old``: identical core, and different when/zone values or negation."""
    sn, so = fact_slots(new), fact_slots(old)
    if not sn.core or sn.core != so.core:
        return False
    if sn.values and so.values and sn.values != so.values:
        return True
    return (sn.negations % 2) != (so.negations % 2)
