# Copyright 2026 STARGA, Inc.
"""Who / when / time-zone slots named in a memory's text.

Text overlap cannot tell "call with Ilona Tue 10am Arizona" from "call with
the contractor Tue 10am Pacific": they share most of their words. These slots
are what makes two look-alike memories different memories (never merge them)
and what makes a corrected memory a changed fact (flag a contradiction).

Deterministic and stateless: regexes and fixed vocabularies, no model, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["FactSlots", "changed_fact", "fact_slots", "slots_differ"]

_WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun".split())
_MONTHS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_ZONES = frozenset(
    "arizona pacific eastern central mountain hawaii alaska atlantic utc gmt pt pst pdt et est edt ct cst cdt "
    "mt mst mdt akst akdt hst cet cest bst ist jst aest aedt".split()
)
# Capitalised words that name nothing (sentence openers, pronouns, articles).
_NOT_NAMES = frozenset(
    "a an the this that these those it its is are was were be i we you he she they me my our your his her their "
    "and or but if then so of in on at to for with by from as not no yes all any each every some new old "
    "use always never note see also please per via".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b|\b(noon|midnight)\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_ZONE_RE = re.compile(r"\b(?:utc|gmt)\s*[+-]\s*\d{1,2}\b", re.I)
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]+")


@dataclass(frozen=True)
class FactSlots:
    """Normalised slot values found in one text."""

    names: frozenset[str]
    days: frozenset[str]
    times: frozenset[str]
    zones: frozenset[str]

    def values(self) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """The slots a correction changes (everything but who)."""
        return self.days, self.times, self.zones


def _times(text: str) -> frozenset[str]:
    out: set[str] = set()
    for m in _TIME_RE.finditer(text):
        if m.group(6):
            out.add(m.group(6).lower())
        elif m.group(3):
            hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
            out.add(f"{hour}:{m.group(2) or '00'}")
        else:
            out.add(f"{int(m.group(4))}:{m.group(5)}")
    return frozenset(out)


def fact_slots(text: str) -> FactSlots:
    """Extract who / day+date / clock time / time zone from *text*."""
    words = set(_WORD_RE.findall(text.lower()))
    calendar = _WEEKDAYS | _MONTHS | _ZONES
    names = {w.lower() for w in _PROPER_RE.findall(text) if w.lower() not in _NOT_NAMES and w.lower() not in calendar}
    days = (words & (_WEEKDAYS | _MONTHS)) | set(_DATE_RE.findall(text))
    zones = (words & _ZONES) | {z.replace(" ", "").lower() for z in _ZONE_RE.findall(text)}
    return FactSlots(frozenset(names), frozenset(days), _times(text), frozenset(zones))


def slots_differ(a: str, b: str) -> bool:
    """True when the texts name different people, days/dates, times or zones."""
    sa, sb = fact_slots(a), fact_slots(b)
    return sa != sb


def changed_fact(new: str, old: str) -> bool:
    """True when *new* restates *old* about the same people with a different day/time/zone.

    Same subject = the same non-empty set of names. A slot counts as changed
    only when both texts state it and the values differ; a text that simply
    omits the time does not contradict one that states it. A correction that
    names both values ("moved to Thursday, no longer Tuesday") still differs.
    """
    sn, so = fact_slots(new), fact_slots(old)
    if not sn.names or sn.names != so.names:
        return False
    return any(x and y and x != y for x, y in zip(sn.values(), so.values()))
