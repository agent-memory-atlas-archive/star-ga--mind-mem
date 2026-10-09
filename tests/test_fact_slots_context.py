# Copyright 2026 STARGA, Inc.
"""Context rules for changed_fact and the merge identity (blind-review findings B1-B4).

Each case is a phrasing a reviewer used to break an earlier version: names that
look like months or weekdays, places that look like zones, identifiers that look
like dates, and corrections written the many ways people write them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.fact_slots import canonical_tokens, changed_fact, slots_differ
from mind_mem.granularity_align import find_merge_candidates, find_similar_pairs, merge_blocks, scan_pairs

# (new, old): different people, places, IDs or events, or an additive change. Not a correction.
NOT_CORRECTIONS = [
    ("Email Jan about the invoice", "Email June about the invoice"),
    ("Ask May about the budget on Tuesday", "Ask April about the budget on Tuesday"),
    ("Lunch with Sun at noon", "Lunch with Wed at noon"),
    ("Alice flies to Arizona at 10am", "Alice flies to Hawaii at 10am"),
    ("Bob is moving to Alaska at 9am", "Bob is moving to Hawaii at 9am"),
    ("Invoice 2026-10-08 is paid", "Invoice 2026-10-09 is paid"),
    ("Ticket 10/8 is closed", "Ticket 10/9 is closed"),
    ("Release 2026-10-08 build is green", "Release 2026-10-09 build is green"),
    ("Ilona's call is on Tuesday", "Ilona's call is on Tuesday and Thursday"),
    ("The call is on Tuesday and Thursday", "The call is on Tuesday"),
    ("The standup is on Tuesday and Thursday.", "The standup is on Tuesday."),
    ("Standup is on Tuesday", "Standup was on Tuesday"),
    ("The March report is late", "The May report is late"),
    ("Book the August conference room", "Book the June conference room"),
    ("Alice was born on Tuesday", "Bob was born on Thursday"),
    ("Meet Alice at 10am Pacific time", "Meet Alice at 10am Pacific"),
    ("Use the 3rd floor room on Tuesday", "Use the 5th floor room on Tuesday"),
    ("Pay the vendor on the 5th floor", "Pay the vendor on the 3rd floor"),
    ("Hire 3 PM interns", "Hire 5 PM interns"),
    ("The standup is at 10 Downing Street", "The standup is at 11 Downing Street"),
    ("The price is 1.50", "The price is 2.50"),
    ("Q3 review on Friday", "Q2 review on Friday"),
    ("Ilona owns INC-2026-10-09.", "Ilona owns INC-2026-10-08."),
    ("Ilona's employee id is 2026-10-09", "Ilona's employee id is 2026-10-08"),
    ("See doc 10/9 for details.", "See doc 10/8 for details."),
    ("Run job 10/8", "Run job 10/9"),
    ("The Thursday Murder Club is on the list.", "The Tuesday Murder Club is on the list."),
    ("The Monday Club meets", "The Friday Club meets"),
    ("Ilona joins Project Tuesday", "Ilona joins Project Thursday"),
    ("Ilona works at April Networks.", "Ilona works at March Networks."),
    ("Meeting in room 10:30 annex", "Meeting in room 11:30 annex"),
    ("Ilona is at 1700 Pennsylvania Ave.", "Ilona is at 1600 Pennsylvania Ave."),
    ("Ilona lives at 1300 Main St.", "Ilona lives at 1200 Main St."),
    ("Deliver to 1700 Amphitheatre Pkwy.", "Deliver to 1600 Amphitheatre Pkwy."),
    (
        "The weekly call with the contractor is on Tuesday at 10am Pacific time.",
        "The weekly call with Ilona is on Tuesday at 10am Arizona time.",
    ),
]

# (new, old): the same fact with a changed time, day, date, zone or polarity.
CORRECTIONS = [
    ("The weekly call with Ilona is on Thursday at 10am Arizona time.", "The weekly call with Ilona is on Tuesday at 10am Arizona time."),
    (
        "The weekly call with Ilona moved to Thursday at 10am Arizona time.",
        "The weekly call with Ilona is on Tuesday at 10am Arizona time.",
    ),
    ("Moved the call from Tuesday to Thursday", "The call is on Tuesday"),
    ("The call is on Thursday rather than Tuesday", "The call is on Tuesday"),
    ("The call is on Thursday instead of Tuesday", "The call is on Tuesday"),
    ("The call is not on Tuesday", "The call is on Tuesday"),
    ("Standup is at 11", "Standup is at 10"),
    ("Standup is at 11 o'clock", "Standup is at 10 o'clock"),
    ("The conference is in 2027", "The conference is in 2026"),
    ("Launch is next Tuesday", "Launch is this Tuesday"),
    ("Launch slipped to Q4", "Launch is in Q3"),
    ("The call is at half past ten", "The call is at ten"),
    ("Standup is at 10.30am", "Standup is at 10am"),
    ("Standup is at 10h30", "Standup is at 10h00"),
    ("Standup moved to 1400", "Standup is at 1000"),
    ("Standup is at 14:00 UTC", "Standup is at 14:00 CET"),
    ("Ilona's call is on Thursday", "Ilona's call is Tuesday"),
    ("Ilona's call is Thursday, not Tuesday", "Ilona's call is on Tuesday"),
    ("Ilona call Thu 10am AZ", "Ilona call Tue 10am AZ"),
    ("Ilona's call is on Thursday.", "Call with Ilona is on Tuesday."),
    ("The call with Ilona moved to Thursday", "Ilona's call is on Tuesday"),
    ("Ilona: call Thursday", "Ilona: call Tuesday"),
    ("We'll meet on Thursday", "We will meet on Tuesday"),
    ("We'll meet the auditors on Thursday.", "We will meet the auditors on Tuesday."),
    ("The deadline is October 9", "The deadline is October 8"),
    ("The deadline is 9 Oct 2026", "The deadline is 8 Oct 2026"),
    ("The deadline is Oct. 9", "The deadline is Oct. 8"),
    ("Rent is due on the 7th of each month", "Rent is due on the 5th of each month"),
    ("Deploys are allowed on Fridays", "Deploys are not allowed on Fridays"),
    ("Do not deploy on Friday", "Deploy on Friday"),
    ("Do not deploy to production on Friday.", "Deploy to production on Friday."),
    ("Alice does not work on Tuesdays", "Alice works on Tuesdays"),
    ("The gym closes at 9pm", "The gym closes at 10pm"),
    ("The gym closes at 21:00", "The gym closes at 22:00"),
    ("The gym closes at 9 PM", "The gym closes at 10 PM"),
    ("The office opens at 8:30 Eastern", "The office opens at 8:30 Central"),
    ("Ilona is in Arizona time", "Ilona is in Pacific time"),
    ("Alice is in Pacific time", "Alice is in Eastern time"),
    ("Ilona's timezone is Arizona", "Ilona's timezone is Pacific"),
    ("Ilona's time zone: MST", "Ilona's time zone: PST"),
    ("Ilona lives in UTC-7", "Ilona lives in UTC-8"),
    ("The 1:1 is Tuesdays at 10", "The 1:1 is Thursdays at 10"),
    ("Standup: Tue→Thu", "Standup: Tue"),
    ("Standup: Tue -> Thu", "Standup: Tue"),
    ("TUESDAY standup", "THURSDAY standup"),
    ("Weekly call with Ilona at 10am (Pacific).", "Weekly call with Ilona at 10am (Arizona)."),
    ("Weekly call with Ilona at 10am - Pacific.", "Weekly call with Ilona at 10am - Arizona."),
    ("Weekly call with Ilona at 10am/Pacific.", "Weekly call with Ilona at 10am/Arizona."),
    ("Weekly call with Ilona at Pacific 10am.", "Weekly call with Ilona at Arizona 10am."),
    ("Weekly call with Ilona at 10am, Pacific.", "Weekly call with Ilona at 10am, Arizona."),
    ("Weekly call with Ilona at 10am [Pacific].", "Weekly call with Ilona at 10am [Arizona]."),
    ("Meet at 1600 tomorrow.", "Meet at 1700 tomorrow."),
    ("Deadline is 2026-10-09T17:00.", "Deadline is 2026-10-08T17:00."),
    ("Ilona does not attend the Tuesday sync.", "Ilona attends the Tuesday sync."),
]


@pytest.mark.parametrize("new,old", NOT_CORRECTIONS)
def test_not_a_correction(new: str, old: str) -> None:
    assert changed_fact(new, old) is False
    assert changed_fact(old, new) is False


@pytest.mark.parametrize("new,old", CORRECTIONS)
def test_correction_flagged(new: str, old: str) -> None:
    assert changed_fact(new, old) is True


@pytest.mark.parametrize("new,old", CORRECTIONS)
def test_correction_against_itself_is_not_flagged(new: str, old: str) -> None:
    # Positive control: the same text never contradicts itself.
    assert changed_fact(new, new) is False
    assert changed_fact(old, old) is False


# ----------------------------------------------------------------------------- merge normal form


@pytest.mark.parametrize(
    "a,b",
    [
        ("Call set_timeout before reconnect.", "Call set _timeout before reconnect."),
        ("Run deploy with --force on prod.", "Run deploy with - -force on prod."),
        ("Logs live in /var/log/app today.", "Logs live in /var/ log/app today."),
        ("Send invoices to ilona@acme.com each month.", "Send invoices to ilona @ acme . com each month."),
        ("The decimal separator is .", "The decimal separator is"),
    ],
)
def test_spacing_around_punctuation_is_content(a: str, b: str) -> None:
    assert slots_differ(a, b)


def test_amount_of_whitespace_and_attached_full_stop_do_not_matter() -> None:
    assert canonical_tokens("Weekly  call\non Tuesday.") == canonical_tokens("Weekly call on Tuesday")


# ----------------------------------------------------------------------------- merge identity


def _blk(bid: str, **fields: object) -> dict:
    return {"_id": bid, "content": "Weekly planning call at 10am.", "Status": "active", **fields}


@pytest.mark.parametrize(
    "field,a,b",
    [
        ("Participants", ["Ilona"], ["the contractor"]),
        ("Attendees", ["Ilona (Arizona)"], ["contractor (Pacific)"]),
        ("When", ["2026-10-06 10am MST"], ["2026-10-08 10am PST"]),
        ("ConstraintSignatures", [{"object": "10am Arizona"}], [{"object": "10am Pacific"}]),
        ("Rationale", "Ilona asked (Arizona time).", "The contractor asked (Pacific time)."),
    ],
)
def test_list_and_mapping_fields_separate_blocks(field: str, a: object, b: object) -> None:
    blocks = [_blk("D-1", **{field: a}), _blk("D-2", **{field: b})]
    assert find_merge_candidates(blocks) == []
    assert [(p.id_a, p.id_b) for p in find_similar_pairs(blocks, min_similarity=0.0)] == [("D-1", "D-2")]
    # Positive control: equal values in the same field DO merge.
    same = [_blk("D-1", **{field: a}), _blk("D-2", **{field: a})]
    assert len(find_merge_candidates(same)) == 1


def test_nested_fields_projection_is_compared_too() -> None:
    a = {"_id": "D-1", "content": "Weekly call.", "fields": {"Participants": ["Ilona"]}}
    b = {"_id": "D-2", "content": "Weekly call.", "fields": {"Participants": ["the contractor"]}}
    assert find_merge_candidates([a, b]) == []


@pytest.mark.parametrize("key", [7, ("a",), 1.5])
def test_non_string_keys_do_not_crash(key: object) -> None:
    a = {"_id": "A", "content": "call ilona tuesday", key: "x"}
    b = {"_id": "B", "content": "call ilona tuesday", key: "x"}
    c = {"_id": "C", "content": "call ilona tuesday", "fields": {key: "y", "Date": "d"}}
    assert len(find_merge_candidates([a, b])) == 1
    candidates, pairs = scan_pairs([a, b, c], min_similarity=0.0)
    assert len(candidates) == 1 and len(pairs) == 2


def test_non_mapping_entries_are_skipped() -> None:
    a = {"_id": "A", "content": "call ilona tuesday"}
    assert len(find_merge_candidates([a, None, dict(a, _id="B"), "text"])) == 1  # type: ignore[list-item]


@pytest.mark.parametrize("status", ["superseded", "archived", "retracted", "Superseded"])
def test_only_active_blocks_merge(status: str) -> None:
    blocks = [_blk("D-1"), _blk("D-2", Status=status)]
    assert find_merge_candidates(blocks) == []
    assert len(find_merge_candidates([_blk("D-1"), _blk("D-2")])) == 1


def test_merge_keeps_both_blocks_provenance() -> None:
    a = _blk("D-1", Sources="email from Ilona 2026-09-01", fields={"Sources": ["email A"]})
    b = _blk("D-2", Sources="email from contractor 2026-09-30", fields={"Sources": ["email B"]})
    (cand,) = find_merge_candidates([a, b])
    merged = merge_blocks(cand.block_a, cand.block_b, strategy=cand.suggested_strategy)
    assert merged["Sources"] == "email from Ilona 2026-09-01; email from contractor 2026-09-30"
    assert merged["fields"]["Sources"] == ["email A", "email B"]


def test_one_pass_returns_both_lists() -> None:
    blocks = [_blk("D-1"), _blk("D-2"), _blk("D-3", Participants=["Ilona"])]
    candidates, pairs = scan_pairs(blocks, min_similarity=0.0)
    assert [(c.block_a["_id"], c.block_b["_id"]) for c in candidates] == [("D-1", "D-2")]
    assert sorted((p.id_a, p.id_b) for p in pairs) == [("D-1", "D-3"), ("D-2", "D-3")]


# ----------------------------------------------------------------------------- end to end


def _workspace(tmp_path: Path, body: str) -> Path:
    (tmp_path / "decisions").mkdir()
    (tmp_path / "mind-mem.json").write_text(
        json.dumps({"version": "5.0.1", "v4": {"granularity_align": {"enabled": True, "min_similarity": 0.5}}})
    )
    (tmp_path / "decisions" / "DECISIONS.md").write_text("# DECISIONS\n\n" + body)
    return tmp_path


def _decision(bid: str, extra: str = "", status: str = "active") -> str:
    return (
        f"[{bid}]\nDate: 2026-10-01\nStatus: {status}\nScope: global\nStatement: Weekly planning call at 10am.\n"
        f"{extra}Rationale: seed.\nSupersedes: none\nTags: test\nSources: test\n\n"
    )


def _plan(ws: Path) -> dict:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.consolidation import plan_consolidation
    from mind_mem.sqlite_index import build_index

    build_index(str(ws))
    with use_workspace(str(ws)):
        return json.loads(plan_consolidation())["granularity_align"]


def test_plan_consolidation_list_participants_not_merged(tmp_path: Path) -> None:
    ws = _workspace(
        tmp_path,
        _decision("D-20261001-001", "Participants:\n- Ilona\n") + _decision("D-20261001-002", "Participants:\n- the contractor\n"),
    )
    g = _plan(ws)
    assert g["scanned_blocks"] == 2  # positive control: both blocks were read
    assert g["candidates"] == []
    assert [(p["id_a"], p["id_b"]) for p in g["similar_pairs"]] == [("D-20261001-001", "D-20261001-002")]


def test_plan_consolidation_exact_duplicates_still_merge(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _decision("D-20261001-001") + _decision("D-20261001-002"))
    g = _plan(ws)
    assert [(c["id_a"], c["id_b"]) for c in g["candidates"]] == [("D-20261001-001", "D-20261001-002")]


def test_plan_consolidation_superseded_copy_not_merged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, _decision("D-20261001-001", status="superseded") + _decision("D-20261001-002"))
    g = _plan(ws)
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


def test_build_index_accepts_list_statement(tmp_path: Path) -> None:
    from mind_mem.sqlite_index import build_index

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\n"
        "Statement:\n- first item of a list statement\n- second item\nContext:\n- a\nTags: t\n"
    )
    result = build_index(str(tmp_path))
    assert result["blocks_indexed"] == 1


def test_contradiction_detector_end_to_end(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    old = "The weekly call with Ilona is on Tuesday at 10am Arizona time."
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        f"# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        f"Statement: {old}\nRationale: seed.\nSupersedes: none\nTags: test\nSources: test\n"
    )

    def kinds(statement: str) -> list[str]:
        return [c["conflict_type"] for c in check_proposal_contradictions(str(tmp_path), {"Statement": statement})["conflicts"]]

    assert "contradiction" in kinds("The weekly call with Ilona is on Thursday at 10am Arizona time.")
    assert "contradiction" not in kinds("Alice flies to Arizona at 10am on Tuesday with Ilona weekly.")
    assert "contradiction" not in kinds("The weekly call with the contractor is on Tuesday at 10am Pacific time.")


# ----------------------------------------------------------------------------- round 3


def test_plan_consolidation_list_statements_not_merged(tmp_path: Path) -> None:
    body = (
        "[D-20261001-001]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement:\n- Weekly call with Ilona Tue 10am Arizona time\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement:\n- Weekly call with the contractor Thu 3pm Pacific time\nTags: test\n\n"
    )
    g = _plan(_workspace(tmp_path, body))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []


@pytest.mark.parametrize(
    "extra_a,extra_b",
    [
        ("speaker: Ilona\n", "speaker: Bob\n"),
        ("Meta:\n  tz: Arizona\n", "Meta:\n  tz: Pacific\n"),
    ],
)
def test_plan_consolidation_unparsed_lines_block_merge(tmp_path: Path, extra_a: str, extra_b: str) -> None:
    g = _plan(_workspace(tmp_path, _decision("D-20261001-001", extra_a) + _decision("D-20261001-002", extra_b)))
    assert g["scanned_blocks"] == 2
    assert g["candidates"] == []
    # Positive control: the same unparsed line on both sides still merges.
    same = tmp_path / "same"
    same.mkdir()
    g2 = _plan(_workspace(same, _decision("D-20261001-001", extra_a) + _decision("D-20261001-002", extra_a)))
    assert len(g2["candidates"]) == 1


def test_parser_keeps_unrecognised_lines() -> None:
    from mind_mem.block_parser import parse_blocks

    (block,) = parse_blocks("[D-20261001-001]\nStatement: x\nspeaker: Bob\nMeta:\n  tz: Pacific\n## Heading\n")
    assert block["_unparsed"] == ["speaker: Bob", "  tz: Pacific"]


@pytest.mark.parametrize(
    "new,old",
    [
        ("Enable the weekly report for Ilona on Tuesday.", "Disable the weekly report for Bob on Tuesday."),
        ("Ilona will start the weekly call on Tuesday at 10am.", "Bob will stop the weekly call on Tuesday at 10am."),
    ],
)
def test_reversal_words_need_the_same_subject(tmp_path: Path, new: str, old: str) -> None:
    from mind_mem.contradiction_detector import _classify_conflict

    assert _classify_conflict(new, old, 0.8) != "contradiction"
    # Positive control: the same subject with the reversal IS a contradiction.
    reversed_same = new.replace("Enable", "Disable").replace("start", "stop")
    assert _classify_conflict(new, reversed_same, 0.8) == "contradiction"


def test_superseded_block_is_not_a_contradiction_candidate(tmp_path: Path) -> None:
    from mind_mem.contradiction_detector import check_proposal_contradictions

    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# DECISIONS\n\n[D-20261001-001]\nDate: 2026-10-01\nStatus: superseded\nScope: global\n"
        "Statement: The weekly call with Ilona is on Tuesday at 10am.\nTags: test\n\n"
        "[D-20261001-002]\nDate: 2026-10-01\nStatus: active\nScope: global\n"
        "Statement: The weekly call with Ilona is on Wednesday at 10am.\nTags: test\n"
    )
    report = check_proposal_contradictions(str(tmp_path), {"Statement": "The weekly call with Ilona is on Thursday at 10am."})
    ids = {c["block_id"] for c in report["conflicts"] if c["conflict_type"] == "contradiction"}
    assert ids == {"D-20261001-002"}  # positive control: the active block is still flagged
