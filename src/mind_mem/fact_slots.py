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


def _spaced_tokens(text: str) -> list[tuple[str, bool]]:
    """(token, preceded_by_whitespace) pairs: word runs (letters, marks, ASCII digits) and single non-space characters."""
    text = unicodedata.normalize("NFC", text)
    out: list[tuple[str, bool]] = []
    cur: list[str] = []
    cur_space = False
    space = False
    n = len(text)
    for i, ch in enumerate(text):
        is_word = ch in _ASCII_DIGITS or unicodedata.category(ch)[0] in ("L", "M")
        if not is_word and ch in _DIGIT_JOINERS and cur and cur[-1] in _ASCII_DIGITS and i + 1 < n and text[i + 1] in _ASCII_DIGITS:
            is_word = True
        if is_word:
            if not cur:
                cur_space = space
            cur.append(ch)
            space = False
            continue
        if cur:
            out.append(("".join(cur), cur_space))
            cur = []
        if ch.isspace():
            space = True
        else:
            out.append((ch, space))
            space = False
    if cur:
        out.append(("".join(cur), cur_space))
    return out


def _raw_tokens(text: str) -> list[str]:
    """Word runs (letters, marks, ASCII digits) and single non-space characters, case preserved."""
    return [tok for tok, _ in _spaced_tokens(text)]


def canonical_tokens(text: str) -> tuple[str, ...]:
    """The MERGE normal form: NFC, and the amount of whitespace does not matter. Nothing else.

    WHETHER whitespace separates two tokens is kept, so "set_timeout" and
    "set _timeout", or "/var/log" and "/var/ log", stay different. A final full
    stop is dropped only when it is attached to a word ("on Tuesday." ==
    "on Tuesday"); a free-standing "." is content ("the separator is .").
    """
    pairs = _spaced_tokens(text)
    if len(pairs) >= 2 and pairs[-1][0] in (".", "。") and not pairs[-1][1] and pairs[-2][0] not in (".", "。"):
        pairs = pairs[:-1]
    return tuple((" " + tok) if (i and space) else tok for i, (tok, space) in enumerate(pairs))


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
#: Abbreviations that are also ordinary words or names ("Sun", "Wed", "Jan",
#: "Mar"): a when-value only in date context. Full weekday names and their
#: plurals are always values.
_WEEKDAY_ABBR = frozenset({"tue", "tues", "thu", "thur", "thurs", "fri", "mon", "wed", "sat", "sun"})
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_MONTHS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
)
_MONTH_ABBR = frozenset(k for k, v in _EQUIV.items() if v in _MONTHS)
#: Zone abbreviations: a value next to a clock or in a time-zone phrase.
_ZONE_ABBR = frozenset("utc gmt pt pst pdt et est edt ct cst cdt mt mst mdt hst akst akdt ist cet cest bst jst aest aedt az".split())
#: Zone NAMES are also places ("flies to Arizona"): a value only right after a
#: clock ("10am Pacific"), before "time" ("Arizona time"), or in a time-zone
#: phrase ("timezone is Arizona").
_ZONE_NAMES = frozenset("pacific eastern central mountain arizona hawaii alaska".split())
_ZONE_CONTEXT = frozenset({"time", "tz", "timezone", "zone"})
_RELATIVE = frozenset("today tomorrow yesterday tonight morning afternoon evening noon midnight weekend".split())
_PERIOD_AFTER = frozenset({"next", "this", "last", "every", "following"})
_PERIODS = frozenset({"week", "month", "year", "weekend"})
#: Words before a month or abbreviated weekday that make it a date ("in May",
#: "by Oct 9", "Standup: Tue") rather than a name ("Ask May", "the May report").
_DATE_PREV = frozenset(
    "from on at in by until till through since before after during early mid late each every next this last following : → ,".split()
)
#: "to" makes a month a date only in a scheduling sentence ("moved the review
#: to March"); otherwise it is usually a recipient ("Send the report to June").
#: "of" counts only after a span word ("end of May"), never in a label
#: ("Review of April numbers").
_SCHEDULING = frozenset(
    "moved move moves moving rescheduled reschedule postponed delayed pushed slipped shifted bumped "
    "due deadline scheduled starts start begins ends launch launches".split()
)
_SPAN_OF = frozenset({"end", "start", "beginning", "middle", "mid", "rest", "first", "last"})
_QUARTER_PREV = frozenset("in by to until till before after during through from since".split())
#: Nouns whose following date-shaped token is an identifier ("Invoice 2026-10-08", "Ticket 10/8").
_ID_NOUNS = frozenset(
    "invoice ticket order build release version ref id case po issue pr bug incident "
    "receipt account room flight item sku batch run job commit tag # no doc document page section chapter "
    "employee badge serial code file row line inc ticket req".split()
)
#: Words that make a following weekday a name ("Project Tuesday", "Operation Monday").
_LABEL_PREV = frozenset({"project", "operation", "codename", "club", "team"})
_ID_GLUE = frozenset({"-", "_", "#", ".", "/"})
_ID_SKIP = frozenset({"is", "was", ":", "=", "no", "number", "#"})
_YEAR_PREV = frozenset("in since by until till from to before after during".split())
# Correction/function words dropped from both cores anywhere.
_CORRECTION = frozenset(
    "moved move moving now instead rescheduled reschedule scheduled changed change switched postponed delayed pushed "
    "slipped slip slips shifted bumped up back got get it is was are were am has have had been be will would do does did "
    "of and but correction actually update updated again starting as the a an with rather than anymore".split()
)
# Prepositions dropped only when a when-value follows ("to Thursday" yes, "to Denver" no).
_VALUE_PREPOSITIONS = frozenset({"to", "from", "for", "on", "at", "in", "by", "until", "till", "through"})
_NEGATION = frozenset({"not", "never", "cancelled", "canceled", "cancel"})
#: State words whose opposite is a negation: "disabled" == "not enabled".
_ANTONYM = {
    "disabled": "enabled",
    "disable": "enable",
    "disallowed": "allowed",
    "inactive": "active",
    "unlocked": "locked",
    "unavailable": "available",
    "unsupported": "supported",
}
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
_CLITICS = {"ll": "will", "re": "are", "ve": "have", "m": "am", "d": "would"}
_MERIDIEM_FORMS = (
    (("a", ".", "m", "."), "am"),
    (("p", ".", "m", "."), "pm"),
    (("a", ".", "m"), "am"),
    (("p", ".", "m"), "pm"),
    (("am",), "am"),
    (("pm",), "pm"),
)
_NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
#: Words that may follow a bare hour ("at 10 on Tuesday", "at ten sharp"); any
#: other word means the number is not a time ("at 10 Downing Street").
_CLOCK_FOLLOWERS = frozenset("on in at for every each daily sharp and or tomorrow today tonight then until till before after".split())
_BARE_HOUR_PREV = frozenset({"at", "until", "till"})
#: "The locker is at 10", "desk at 12": a number after "at" that locates a thing is not an hour.
_PLACE_NOUNS = frozenset(
    "locker desk room seat gate table platform booth slot stand berth bay dock pier lane row aisle "
    "cabin bed office suite unit apartment apt floor level stall pump terminal".split()
)
#: Ratios, scores and references written with a colon are not clock times.
_NOT_TIME_NOUNS = frozenset("ratio score odds verse chapter version aspect result tally".split())
_HHMM_PREV = frozenset({"at", "to", "until", "till", "by", "from"})
_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(t\d{1,2}:\d{2}(:\d{2})?z?)?$")
_SLASH_DATE = re.compile(r"^\d{1,4}/\d{1,2}(/\d{1,4})?$")
_CLOCK = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?(am|pm)?$")
_H_CLOCK = re.compile(r"^(\d{1,2})h(\d{2})$")
_CANON_CLOCK = re.compile(r"^t\d{2}:\d{2}a?$")
_ORDINAL_DAY = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_OFFSET_ZONE = re.compile(r"^(utc|gmt)[+-]\d{1,2}(:\d{2})?$")
_QUARTER = re.compile(r"^q[1-4]$")
_APOSTROPHES = ("'", "’")
_SPECIAL_PUNCT = frozenset({"'", "’", "-", "=", "+", "−", "→", "⇒", "⟶"})
_ARROWS = ("→", "⇒", "⟶")


def _canon_clock(hour: int, minute: int | None, meridiem: str | None, *, twenty_four: bool = False) -> str | None:
    m = minute or 0
    if m > 59:
        return None
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        return f"t{hour % 12 + (12 if meridiem == 'pm' else 0):02d}:{m:02d}"
    if hour > 23:
        return None
    # A bare "7:00" or "at 7" does not say am or pm: it stays distinct from both ("a" = ambiguous).
    return f"t{hour:02d}:{m:02d}" if (twenty_four or hour > 12) else f"t{hour:02d}:{m:02d}a"


def _time_word(tok: str) -> bool:
    return tok in _ZONE_ABBR or tok in _ZONE_NAMES or tok in _RELATIVE or _EQUIV.get(tok, tok) in _WEEKDAYS


def _follower_ok(raw: list[str], j: int) -> bool:
    """True when what follows a bare number lets it be an hour (end, punctuation, a time word).

    "at 14, Elm Street" and "at 12 in building B" are addresses: a comma or
    "in" counts only when a time word follows ("at 10, tomorrow", "at 10 in
    the morning").
    """
    n = len(raw)
    if j >= n:
        return True
    nxt = raw[j]
    if nxt in (",", "in"):
        k = j + 1
        while k < n and raw[k] in ("the", "a"):
            k += 1
        if k >= n:
            return True
        after = raw[k]
        if not after.isalpha():
            return nxt == ","
        return _time_word(after) or after in _CLOCK_FOLLOWERS
    if not nxt.isalpha():
        return True
    return nxt in _CLOCK_FOLLOWERS or _time_word(nxt)


def _plural_noun(tok: str) -> bool:
    """A counted noun after "3 PM" ("3 PM interns" is a head count, not a time)."""
    return tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and tok not in _ZONE_ABBR


def _compare_tokens(text: str) -> list[str]:
    return _compare_tokens_cased(text)[0]


def _compare_tokens_cased(text: str) -> tuple[list[str], list[bool]]:
    """Normal form for changed_fact: case-folded; contractions, clocks, arrows and offsets resolved.

    Abbreviated month and weekday names are kept as written here: whether one
    is a date or a name is decided in context by :func:`_value_at`.
    """
    orig = _raw_tokens(text)
    raw = [t.casefold() for t in orig]
    out: list[str] = []
    titled: list[bool] = []  # was the source token Title-case ("Networks", "Club")?
    i, n = 0, len(raw)
    while i < n:
        mark = len(out)
        tok = raw[i]
        if len(tok) == 1 and not tok.isalnum() and tok not in _SPECIAL_PUNCT:
            out.append(tok)  # plain punctuation: nothing below can apply (fast path)
            titled.append(False)
            i += 1
            continue
        prev = out[-1] if out else ""
        if tok in _CONTRACTION_BASE and i + 2 < n and raw[i + 1] in _APOSTROPHES and raw[i + 2] == "t":
            out.extend((_CONTRACTION_BASE[tok], "not"))
            i += 3
            titled.extend([False] * (len(out) - mark))
            continue
        if tok == "cannot":
            out.extend(("can", "not"))
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _APOSTROPHES and i + 1 < n and raw[i + 1] == "s":
            i += 2  # "call's" == "call is"; a possessive "Ilona's" drops the same way on both sides
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _APOSTROPHES and i + 1 < n and raw[i + 1] in _CLITICS and out and out[-1].isalpha():
            out.append(_CLITICS[raw[i + 1]])  # "We'll" == "We will"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in ("-", "=") and i + 1 < n and raw[i + 1] == ">":
            out.append("→")
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _ARROWS:
            out.append("→")
            i += 1
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in ("utc", "gmt") and i + 2 < n and raw[i + 1] in ("+", "-", "−") and raw[i + 2].isdigit():
            out.append(f"{tok}{'+' if raw[i + 1] == '+' else '-'}{int(raw[i + 2])}")
            i += 3
            titled.extend([False] * (len(out) - mark))
            continue
        if tok in _MONTH_ABBR and i + 1 < n and raw[i + 1] == ".":
            out.append(tok)  # "Oct. 9" == "Oct 9"
            i += 2
            titled.extend([False] * (len(out) - mark))
            continue
        clock, used = _clock_at(raw, i, prev)
        if clock is not None:
            out.append(clock)
            i += used
            titled.extend([False] * (len(out) - mark))
            continue
        out.append(tok)
        i += 1
        titled.extend([orig[i - 1].istitle() and not orig[i - 1].isupper()] * (len(out) - mark))
        continue
    return out, titled


def _clock_at(raw: list[str], i: int, prev: str) -> tuple[str | None, int]:
    """A canonical clock starting at raw[i] and the number of tokens it uses, or (None, 0)."""
    tok = raw[i]
    n = len(raw)
    # "half past ten", "quarter to 11"
    if tok in ("half", "quarter") and i + 2 < n and raw[i + 1] in ("past", "to"):
        hour = _NUMBER_WORDS.get(raw[i + 2], int(raw[i + 2]) if _is_digits(raw[i + 2]) else None)
        if hour is not None and 1 <= hour <= 12:
            minute = 30 if tok == "half" else 15
            if raw[i + 1] == "to":
                if tok == "half":
                    return None, 0
                hour, minute = (hour - 1) or 12, 45
            return _canon_clock(hour, minute, None), 3
    m = _H_CLOCK.match(tok)
    if m:
        return _canon_clock(int(m.group(1)), int(m.group(2)), None, twenty_four=True), 1
    hour_word = _NUMBER_WORDS.get(tok)
    if hour_word is not None and 1 <= hour_word <= 12:
        if i + 3 < n + 1 and raw[i + 1 : i + 4] == ["o", "'", "clock"]:
            return _canon_clock(hour_word, None, None), 4
        if prev in _BARE_HOUR_PREV and _follower_ok(raw, i + 1):
            return _canon_clock(hour_word, None, None), 1
        return None, 0
    if prev in _ID_NOUNS:
        return None, 0  # "room 10:30", "ticket 1400": identifiers, not times
    if not any(c in _ASCII_DIGITS for c in tok) and tok not in _NUMBER_WORDS and tok not in ("half", "quarter"):
        return None, 0
    back = [raw[k] for k in range(max(0, i - 5), i)]
    if any(t in _NOT_TIME_NOUNS for t in back[-3:]):
        return None, 0  # "the ratio is 16:10", "score was 21:15"
    if prev in _BARE_HOUR_PREV and _is_digits(tok) and any(t in _PLACE_NOUNS for t in back):
        return None, 0  # "Ilona's locker is at 10": a bare number locating a thing, not an hour
    if _is_digits(tok) and len(tok) == 4 and prev in _HHMM_PREV:
        value = int(tok)
        is_year = 1900 <= value <= 2099 and prev != "at"
        hours = i + 1 < n and raw[i + 1] in ("hours", "hrs")
        # "at 1600 Pennsylvania Ave" is an address: a word after the number rules out a time.
        if not is_year and value // 100 < 24 and value % 100 < 60 and (hours or _follower_ok(raw, i + 1)):
            used = 2 if i + 1 < n and raw[i + 1] in ("hours", "hrs") else 1
            return _canon_clock(value // 100, value % 100, None, twenty_four=True), used
    m = _CLOCK.match(tok)
    if not m:
        return None, 0
    hour, minute_s, mer = int(m.group(1)), m.group(2), m.group(3)
    j = i + 1
    if mer is None:
        for form, value in _MERIDIEM_FORMS:
            if tuple(raw[j : j + len(form)]) == form:
                mer, j = value, j + len(form)
                break
    if minute_s is not None and "." in tok and mer is None:
        return None, 0  # "1.50" is a number, "10.30am" a time
    if raw[j : j + 3] == ["o", "'", "clock"] and minute_s is None and mer is None:
        return _canon_clock(hour, None, None), j + 3 - i
    if mer is not None:
        if minute_s is None and j < n and _plural_noun(raw[j]):
            return None, 0  # "Hire 3 PM interns"
        return _canon_clock(hour, int(minute_s) if minute_s else None, mer), j - i
    if minute_s is not None:
        return _canon_clock(hour, int(minute_s), None), 1
    if prev in _BARE_HOUR_PREV and _follower_ok(raw, i + 1):
        return _canon_clock(hour, None, None), 1  # "The standup is at 10"
    return None, 0


@dataclass(frozen=True)
class FactSlots:
    """A statement split into its when/zone values and everything else (the core)."""

    core: tuple[str, ...]
    values: frozenset[str]
    negations: int


def _is_clock(tok: str) -> bool:
    return bool(_CANON_CLOCK.match(tok))


def _near_clock(toks: list[str], i: int, span: int = 2) -> bool:
    return any(_is_clock(toks[k]) for k in range(max(0, i - span), min(len(toks), i + span + 1)) if k != i)


def _zone_phrase_before(toks: list[str], i: int) -> bool:
    return any(toks[k] in _ZONE_CONTEXT for k in range(max(0, i - 3), i))


def _date_context(toks: list[str], i: int) -> bool:
    """Is the month/abbreviated weekday at *i* used as a date?"""
    prev = toks[i - 1] if i > 0 else ""
    nxt = toks[i + 1] if i + 1 < len(toks) else ""

    def dayish(t: str) -> bool:
        return _is_digits(t) or bool(_ORDINAL_DAY.match(t)) or _is_clock(t) or bool(_SLASH_DATE.match(t) or _ISO_DATE.match(t))

    if prev in _DATE_PREV or dayish(prev) or dayish(nxt) or nxt == "→":
        return True
    if prev == "to":
        return any(t in _SCHEDULING for t in toks)
    if prev == "of":
        return i >= 2 and toks[i - 2] in _SPAN_OF
    if prev in ("and", "or", "&", "/") and i >= 2:
        before = toks[i - 2]  # "Tue and Thu": a list continues the date it follows
        return before in _WEEKDAYS or before in _WEEKDAY_ABBR or before in _MONTHS or before in _MONTH_ABBR
    return False


def _id_context(toks: list[str], i: int) -> bool:
    """Is the date-shaped token at *i* part of an identifier ("INC-2026-10-09", "employee id is 2026-10-08")?"""
    prev = toks[i - 1] if i > 0 else ""
    if prev in _ID_GLUE and i >= 2 and toks[i - 2].isalnum() and not _is_digits(toks[i - 2]):
        return True
    k = i - 1
    while k >= 0 and toks[k] in _ID_SKIP and i - k <= 3:
        k -= 1
    return k >= 0 and (toks[k] in _ID_NOUNS or prev in _ID_NOUNS)


def _named(toks: list[str], titled: list[bool], i: int) -> bool:
    """Is the day/month word at *i* part of a proper name ("March Networks", "The Thursday Murder Club", "Project Tuesday")?"""
    prev = toks[i - 1] if i > 0 else ""
    if prev in _LABEL_PREV:
        return True
    j = i + 1
    if j < len(toks) and titled[j] and toks[j].isalpha():
        nxt = toks[j]
        date_word = nxt in _WEEKDAYS or nxt in _MONTHS or nxt in _ZONE_NAMES or nxt in _ZONE_ABBR or nxt in _RELATIVE or nxt in _EQUIV
        return not date_word and nxt not in _ZONE_CONTEXT
    return False


def _zone_name_by_clock(toks: list[str], i: int) -> bool:
    """A zone name whose nearest word on either side, skipping punctuation, is a clock ("10am (Pacific)", "Pacific 10am")."""
    for step in (-1, 1):
        k = i + step
        while 0 <= k < len(toks) and len(toks[k]) == 1 and not toks[k].isalnum():
            k += step
        if 0 <= k < len(toks) and _is_clock(toks[k]):
            return True
    return False


def _value_at(toks: list[str], i: int, titled: list[bool] | None = None) -> str | None:
    """The when/zone value at position *i*, judged in context, or None."""
    titled = titled if titled is not None else [False] * len(toks)
    t = toks[i]
    if len(t) == 1 and not t.isalnum():
        return None  # punctuation and symbols are never values (fast path)
    n = len(toks)
    prev = toks[i - 1] if i > 0 else ""
    nxt = toks[i + 1] if i + 1 < n else ""
    if _is_clock(t) and t.endswith("a") and i >= 2 and titled[i - 1] and prev.isalpha() and not _time_word(prev):
        return None  # "John 3:16", "Genesis 1:10": a reference after a name, not a time
    if _is_clock(t) or t in _RELATIVE or _OFFSET_ZONE.match(t):
        return t
    if _QUARTER.match(t):
        return t if prev in _QUARTER_PREV else None  # "slipped to Q4" yes, "the review for Q3" no
    if (t in _WEEKDAYS or t in _MONTHS or t in _EQUIV) and _named(toks, titled, i):
        return None
    if t in _WEEKDAYS or (t in _EQUIV and _EQUIV[t] in _WEEKDAYS and t not in _WEEKDAY_ABBR):
        day = _EQUIV.get(t, t)
        return f"{prev}-{day}" if prev in ("next", "this", "last", "following") else day
    if t in _WEEKDAY_ABBR:
        return _EQUIV[t] if _date_context(toks, i) else None
    if t in _MONTHS or t in _MONTH_ABBR:
        return _EQUIV.get(t, t) if _date_context(toks, i) else None
    if _ISO_DATE.match(t) or _SLASH_DATE.match(t):
        return None if _id_context(toks, i) else "d" + t
    if t in _ZONE_ABBR:
        return t if (_near_clock(toks, i) or nxt in _ZONE_CONTEXT or _zone_phrase_before(toks, i)) else None
    if t in _ZONE_NAMES:
        if i + 1 < n and titled[i + 1] and nxt.isalpha() and not _time_word(nxt) and nxt not in _ZONE_CONTEXT:
            return None  # "6am Pacific Surfliner": part of a proper name
        return t if (_zone_name_by_clock(toks, i) or nxt in _ZONE_CONTEXT or _zone_phrase_before(toks, i)) else None
    if t in _PERIODS and prev in _PERIOD_AFTER:
        return prev + "-" + t
    if _is_digits(t) and len(t) == 4 and 1900 <= int(t) <= 2099 and (prev in _YEAR_PREV or prev in _MONTHS or prev in _MONTH_ABBR):
        return "y" + t
    od = _ORDINAL_DAY.match(t)
    month_near = prev in _MONTHS or prev in _MONTH_ABBR or nxt in _MONTHS or nxt in _MONTH_ABBR
    if (_is_digits(t) or od) and len(t) <= 4 and month_near:
        return "n" + (od.group(1) if od else t)
    if od and prev == "the" and (nxt in ("", "of", "day") or len(nxt) == 1 and not nxt.isalnum()):
        return "n" + od.group(1)
    return None


def _stem(tok: str) -> str:
    """Light plural/3rd-person fold for core comparison ("works" == "work", "deploys" == "deploy")."""
    return tok[:-1] if tok.isalpha() and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def fact_slots(text: str) -> FactSlots:
    toks, titled = _compare_tokens_cased(text)
    n = len(toks)
    found = [_value_at(toks, i, titled) for i in range(n)]
    is_value = [v is not None for v in found]
    for i, v in enumerate(found):
        if v is not None and i > 0 and (toks[i] in _PERIODS or v.startswith(toks[i - 1] + "-")):
            is_value[i - 1] = True  # "next" in "next week" / "next Tuesday" is part of the value

    def prev_word(i: int) -> int:
        k = i - 1
        while k >= 0 and toks[k] in ("the", "a", "an", "on", "at", "in"):
            k -= 1
        return k

    values: set[str] = set()
    rejected: list[tuple[int, str]] = []
    for i, v in enumerate(found):
        if v is None:
            continue
        k = prev_word(i)
        if k >= 0 and toks[k] == "from":
            continue  # "from Tuesday to Thursday": Tuesday is the old value
        if k >= 1 and (toks[k], toks[k - 1]) in (("of", "instead"), ("than", "rather")):
            continue  # "Thursday instead of Tuesday": Tuesday is the old value
        if i + 1 < n and toks[i + 1] == "→":
            continue  # "Tue → Thu": the value before the arrow is the old one
        if k >= 0 and toks[k] in _NEGATION:
            rejected.append((k, v))
            continue
        values.add(v)
    skipped_negation: set[int] = set()
    if values:
        # "Thursday, not Tuesday": a contrast names the rejected value; it is not a negated fact.
        skipped_negation = {k for k, _ in rejected}
    else:
        # "The call is not on Tuesday": the negation applies to the fact itself.
        values.update(v for _, v in rejected)
    core: list[str] = []
    negations = 0
    for i, t in enumerate(toks):
        if is_value[i] or t == "→":
            continue
        if t in _NEGATION or (t == "no" and i + 1 < n and toks[i + 1] == "longer"):
            if i not in skipped_negation:
                negations += 1
            continue
        if t == "longer" and i > 0 and toks[i - 1] == "no":
            continue
        if (t == "from" and i + 1 < n and toks[i + 1] == "now") or (t == "on" and i > 0 and toks[i - 1] == "now"):
            continue  # "from now on"
        if t in _CORRECTION:
            continue
        if t in _ANTONYM:
            negations += 1
            core.append(_stem(_ANTONYM[t]))
            continue
        if t in _ZONE_CONTEXT and i > 0 and is_value[i - 1] and (toks[i - 1] in _ZONE_ABBR or toks[i - 1] in _ZONE_NAMES):
            continue  # "Arizona time"
        if t in _ZONE_CONTEXT and any(is_value[k] for k in range(i + 1, min(n, i + 4))):
            continue  # "timezone is Arizona"
        if t in _VALUE_PREPOSITIONS:
            k = i + 1
            while k < n and toks[k] in ("the", "a", "an"):
                k += 1
            if k < n and is_value[k]:
                continue  # "to Thursday", "from Tuesday", "on the 5th"
        if len(t) == 1 and unicodedata.category(t)[0] == "P":
            continue  # punctuation; symbols ($, €, +, <, emoji) stay in the core
        core.append(_stem(t))
    return FactSlots(core=tuple(sorted(core)), values=frozenset(values), negations=negations)


def changed_fact(new: str, old: str) -> bool:
    """True when ``new`` corrects ``old``.

    The cores (everything but when/zone values and correction words) must hold
    the same words, in any order ("Ilona's call" == "call with Ilona"). Then
    either the value sets differ, without one merely adding to the other
    ("Tuesday" -> "Tuesday and Thursday" is a refinement), or the negation flips.
    """
    sn, so = fact_slots(new), fact_slots(old)
    if not sn.core or sn.core != so.core:
        return False
    if sn.values and so.values and sn.values != so.values:
        if not (sn.values < so.values or so.values < sn.values):
            return True
    return (sn.negations % 2) != (so.negations % 2)
