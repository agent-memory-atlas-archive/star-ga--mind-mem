# Copyright 2026 STARGA, Inc.
"""Fail-closed fact comparison for memory merges and contradictions.

Two decisions read this module, built so that words alone can decide them safely:

* ``slots_differ(a, b)`` / ``canonical_tokens`` — may two memories be MERGED?
  Only when they are the same statement up to Unicode NFC, whitespace and a
  trailing full stop. Case, spelling, abbreviations and number/time formats
  must match exactly: "US" vs "us", "Jun" vs "June", "7:00" vs "7am" can each
  mean different things, and a wrongly proposed merge loses a memory.
* ``changed_fact(new, old)`` — is ``new`` a correction of ``old``? Both texts
  are normalised for comparison (case, unambiguous abbreviations, canonical
  clock times, contractions), the when/zone values and correction words are
  removed IN CONTEXT, and what remains (people, event, identifiers, places,
  quantities) must be identical; then the value sets must differ or the
  negation flip. The context rules keep different people, places, teams and
  IDs from being flagged.

Both decisions are linear in the text length. Only ASCII 0-9 count as digits.
Deterministic, no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["FactSlots", "canonical_tokens", "changed_fact", "fact_slots", "slots_differ"]

_ASCII_DIGITS = frozenset("0123456789")
# Kept inside a token when an ASCII digit is directly on both sides: "1.5", "1,000", "10:30", "2026-10-08", "10/8".
_DIGIT_JOINERS = frozenset(".,:-/")


def _is_digits(tok: str) -> bool:
    return bool(tok) and all(c in _ASCII_DIGITS for c in tok)


def _raw_tokens(text: str) -> list[str]:
    """Word runs (letters, marks, ASCII digits) and single non-space characters, case preserved."""
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
            out.append("".join(cur))
            cur = []
        if not ch.isspace():
            out.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def canonical_tokens(text: str) -> tuple[str, ...]:
    """The MERGE normal form: NFC, whitespace-insensitive, trailing full stop dropped. Nothing else."""
    out = _raw_tokens(text)
    while out and out[-1] in (".", "。"):
        out.pop()
    return tuple(out)


def slots_differ(a: str, b: str) -> bool:
    """True unless the two texts are the same statement (see :func:`canonical_tokens`)."""
    return canonical_tokens(a) != canonical_tokens(b)


# ----------------------------------------------------------------------------- changed_fact

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
    "mon": "monday",
    "mondays": "monday",
    "wed": "wednesday",
    "wednesdays": "wednesday",
    "sat": "saturday",
    "saturdays": "saturday",
    "sun": "sunday",
    "sundays": "sunday",
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sept": "september",
    "sep": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
)
_ZONES = frozenset(
    "utc gmt pt pst pdt et est edt ct cst cdt mt mst mdt hst akst akdt ist cet cest bst jst aest aedt "
    "pacific eastern central mountain arizona hawaii alaska".split()
)
_ZONE_CONTEXT = frozenset({"time", "tz", "timezone", "zone"})
_RELATIVE = frozenset("today tomorrow yesterday tonight morning afternoon evening noon midnight weekend".split())
_PERIOD_AFTER = frozenset({"next", "this", "last", "every", "following"})
_PERIODS = frozenset({"week", "month", "year", "weekend"})
# Correction/function words dropped from both cores anywhere.
_CORRECTION = frozenset(
    "moved move moving now instead rescheduled reschedule scheduled changed change switched postponed delayed pushed "
    "up back got get it is was are were has have had been be will of and but correction actually update updated "
    "again starting as the a an".split()
)
# Prepositions dropped only when a when-value follows ("to Thursday" yes, "to Denver" no).
_VALUE_PREPOSITIONS = frozenset({"to", "from", "for", "on", "at", "in", "by", "until", "till", "through"})
_NEGATION = frozenset({"not", "never"})
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
_MERIDIEM_FORMS = (
    (("a", ".", "m", "."), "am"),
    (("p", ".", "m", "."), "pm"),
    (("a", ".", "m"), "am"),
    (("p", ".", "m"), "pm"),
    (("am",), "am"),
    (("pm",), "pm"),
)
_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_SLASH_DATE = re.compile(r"^\d{1,4}/\d{1,2}(/\d{1,4})?$")
_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$")
_CANON_CLOCK = re.compile(r"^t\d{2}:\d{2}a?$")
_ORDINAL_DAY = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_APOSTROPHES = ("'", "’")


def _canon_clock(hour: str, minute: str | None, meridiem: str | None) -> str | None:
    h = int(hour)
    m = int(minute) if minute else 0
    if m > 59:
        return None
    if meridiem:
        if not 1 <= h <= 12:
            return None
        return f"t{h % 12 + (12 if meridiem == 'pm' else 0):02d}:{m:02d}"
    if minute is None or h > 23:
        return None
    # A bare "7:00" does not say am or pm: it stays distinct from both ("a" = ambiguous).
    return f"t{h:02d}:{m:02d}a" if h <= 12 else f"t{h:02d}:{m:02d}"


def _compare_tokens(text: str) -> list[str]:
    """Normal form for changed_fact: case-folded; abbreviations, clocks and contractions resolved."""
    raw = [t.casefold() for t in _raw_tokens(text)]
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        tok = raw[i]
        if tok in _CONTRACTION_BASE and i + 2 < n and raw[i + 1] in _APOSTROPHES and raw[i + 2] == "t":
            out.extend((_CONTRACTION_BASE[tok], "not"))
            i += 3
            continue
        if tok == "cannot":
            out.extend(("can", "not"))
            i += 1
            continue
        if tok in _APOSTROPHES and i + 1 < n and raw[i + 1] == "s":
            i += 2  # "call's" == "call is"; a possessive "Ilona's" drops the same way on both sides
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
        out.append(_EQUIV.get(tok, tok))
        i += 1
    return out


@dataclass(frozen=True)
class FactSlots:
    """A statement split into its when/zone values and everything else (the core)."""

    core: tuple[str, ...]
    values: frozenset[str]
    negations: int


def _value_at(toks: list[str], i: int) -> str | None:
    """The when/zone value at position *i*, judged in context, or None."""
    t = toks[i]
    n = len(toks)
    prev = toks[i - 1] if i > 0 else ""
    nxt = toks[i + 1] if i + 1 < n else ""
    if _CANON_CLOCK.match(t) or t in _WEEKDAYS or t in _MONTHS or t in _RELATIVE:
        return t
    if _ISO_DATE.match(t) or _SLASH_DATE.match(t):
        return "d" + t
    if t in _ZONES:
        near_clock = any(_CANON_CLOCK.match(toks[k]) for k in range(max(0, i - 2), min(n, i + 3)))
        return t if (near_clock or nxt in _ZONE_CONTEXT) else None
    if t in _PERIODS and prev in _PERIOD_AFTER:
        return prev + "-" + t
    od = _ORDINAL_DAY.match(t)
    if (_is_digits(t) or od) and (prev in _MONTHS or nxt in _MONTHS):
        return "n" + (od.group(1) if od else t)
    if od and prev == "the" and (nxt in ("", "of", "day") or nxt in _MONTHS or (len(nxt) == 1 and not nxt.isalnum())):
        return "n" + od.group(1)
    return None


def fact_slots(text: str) -> FactSlots:
    toks = _compare_tokens(text)
    n = len(toks)
    values: set[str] = set()
    is_value = [False] * n
    for i in range(n):
        v = _value_at(toks, i)
        if v is not None:
            values.add(v)
            is_value[i] = True
            if toks[i] in _PERIODS and i > 0:
                is_value[i - 1] = True  # "next" in "next week" is part of the value
    core: list[str] = []
    negations = 0
    for i, t in enumerate(toks):
        if is_value[i]:
            continue
        if t in _NEGATION or (t == "no" and i + 1 < n and toks[i + 1] == "longer"):
            negations += 1
            continue
        if t == "longer" and i > 0 and toks[i - 1] == "no":
            continue
        if (t == "from" and i + 1 < n and toks[i + 1] == "now") or (t == "on" and i > 0 and toks[i - 1] == "now"):
            continue  # "from now on"
        if t in _CORRECTION:
            continue
        if t in _ZONE_CONTEXT and i > 0 and is_value[i - 1] and toks[i - 1] in _ZONES:
            continue  # "Arizona time"
        if t in _VALUE_PREPOSITIONS:
            k = i + 1
            while k < n and toks[k] in ("the", "a", "an"):
                k += 1
            if k < n and is_value[k]:
                continue  # "to Thursday", "from Tuesday", "on the 5th"
        if len(t) == 1 and unicodedata.category(t)[0] == "P":
            continue  # punctuation; symbols ($, €, +, <, emoji) stay in the core
        core.append(t)
    return FactSlots(core=tuple(core), values=frozenset(values), negations=negations)


def changed_fact(new: str, old: str) -> bool:
    """True when ``new`` corrects ``old``: identical core, and different when/zone values or a negation flip."""
    sn, so = fact_slots(new), fact_slots(old)
    if not sn.core or sn.core != so.core:
        return False
    if sn.values and so.values and sn.values != so.values:
        return True
    return (sn.negations % 2) != (so.negations % 2)
