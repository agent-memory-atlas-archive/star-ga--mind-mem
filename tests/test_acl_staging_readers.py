"""A user-scope staging write is allowed only while no reader serves it to others.

``USER_STAGING_WRITES`` lets four user-scope tools append to their own store.
That is only safe while nothing reads the store back into what OTHER clients
are served. ``recall_outcome`` broke the rule: with the supported opt-in
``recall.validity_gate.outcome_attribution`` on, every client's recall reads it
and reweights its served scores, so one user-scope ``report_outcome`` could
demote a block for everyone -- and the gate checked only v4 flags.

The fix derives the gate from the READERS of each staged store
(``STAGING_STORE_READERS``). This module holds that table to the code:

* an AST scan finds every function that touches a staged store, directly or
  through another function that does, and fails on any it finds unclassified --
  so a new reader (and the config key that turns it on) cannot be forgotten;
* a live reader that serves other clients makes the write admin-scope, proved
  end to end with the positive control (admin writes) and the mutation (reader
  off -> the user write is staging again).
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from typing import Any, Iterator

import pytest

from mind_mem.mcp.infra.acl import (
    ADMIN_TOOLS,
    STAGING_STORE_READERS,
    USER_STAGING_WRITES,
    check_write_mode_acl,
    cross_client_reader_config_keys,
    staging_store_token,
)
from mind_mem.mcp.infra.workspace import use_workspace

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
#: The classifier itself names every store; scanning it would be circular.
_EXCLUDED_MODULES = frozenset({"mind_mem.mcp.infra.acl"})


# ---------------------------------------------------------------------------
# The scan: every function that touches a staged store, transitively
# ---------------------------------------------------------------------------


class _Source:
    """Parsed ``src/mind_mem`` with just enough name resolution for the scan."""

    def __init__(self) -> None:
        self.text: dict[str, str] = {}
        self.funcs: dict[str, ast.AST] = {}  # "module.Qual.name" -> node
        self.module_of: dict[str, str] = {}
        self.class_methods: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.by_leaf: dict[str, set[str]] = defaultdict(set)
        for path in sorted(SRC.rglob("*.py")):
            parts = path.relative_to(SRC).with_suffix("").parts
            mod = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            if mod in _EXCLUDED_MODULES:
                continue
            text = path.read_text(encoding="utf-8")
            self.text[mod] = text
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add(mod, node.name, node)
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._add(mod, f"{node.name}.{item.name}", item)
                            self.class_methods[(mod, node.name)].add(item.name)
            self._imports(mod, tree, path.name == "__init__.py")

    def _add(self, mod: str, qual: str, node: ast.AST) -> None:
        key = f"{mod}.{qual}"
        self.funcs[key] = node
        self.module_of[key] = mod
        self.by_leaf[qual.rsplit(".", 1)[-1]].add(key)

    def _imports(self, mod: str, tree: ast.AST, is_pkg: bool) -> None:
        table: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pkg = mod.split(".") if is_pkg else mod.split(".")[:-1]
                    pkg = pkg[: len(pkg) - (node.level - 1)]
                    base = ".".join(pkg + ([base] if base else []))
                for alias in node.names:
                    table[alias.asname or alias.name] = (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    table[alias.asname or alias.name.split(".")[0]] = (alias.name if alias.asname else alias.name.split(".")[0], None)
        self._import_table = getattr(self, "_import_table", {})
        self._import_table[mod] = table

    def _resolve_name(self, mod: str, name: str) -> str | None:
        """``name`` as seen in *mod* -> "module.Qual" (a function or a class)."""
        imported = self._import_table[mod].get(name)
        if imported and imported[1]:
            return f"{imported[0]}.{imported[1]}"
        return f"{mod}.{name}"

    def references(self, key: str) -> set[str]:
        """Function keys *key* may call: names resolved through imports, methods
        through ``Class(...)``/``var = Class(...)``/``self``, else a leaf name
        when every definition of it is a candidate (decided by the caller)."""
        mod = self.module_of[key]
        node = self.funcs[key]
        owner = key[len(mod) + 1 :].split(".")[0] if "." in key[len(mod) + 1 :] else None
        local_cls: dict[str, str] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call) and isinstance(sub.value.func, ast.Name):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        local_cls[target.id] = self._resolve_name(mod, sub.value.func.id) or ""
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out.add(self._resolve_name(mod, sub.id) or "")
            elif isinstance(sub, ast.Attribute):
                value = sub.value
                cls: str | None = None
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    cls = self._resolve_name(mod, value.func.id)
                elif isinstance(value, ast.Name) and value.id == "self" and owner:
                    cls = f"{mod}.{owner}"
                elif isinstance(value, ast.Name) and value.id in local_cls:
                    cls = local_cls[value.id]
                elif isinstance(value, ast.Name) and self._import_table[mod].get(value.id, ("", "x"))[1] is None:
                    out.add(f"{self._import_table[mod][value.id][0]}.{sub.attr}")
                    continue
                if cls:
                    out.add(f"{cls}.{sub.attr}")
                else:
                    out.add(f"*.{sub.attr}")
        return out

    def touching(self, token: str, terminals: frozenset[str]) -> set[str]:
        """Every function reaching *token*, stopping at classified terminals."""
        pattern = re.compile(rf"\b{re.escape(token)}\b")
        found = {k for k, n in self.funcs.items() if pattern.search(ast.get_source_segment(self.text[self.module_of[k]], n) or "")}
        refs = {k: self.references(k) for k in self.funcs}
        while True:
            live = {k for k in found if k not in terminals}
            names = set(live)
            for k in live:
                leaf = k.rsplit(".", 1)[-1]
                if self.by_leaf[leaf] <= found:
                    names.add(f"*.{leaf}")
            grown = {k for k, r in refs.items() if k not in found and r & names}
            if not grown:
                return found
            found |= grown


@pytest.fixture(scope="module")
def source() -> _Source:
    return _Source()


def _terminals(token: str) -> frozenset[str]:
    return frozenset(k for k, r in STAGING_STORE_READERS[token].items() if r.role != "plumbing")


@pytest.mark.parametrize("token", sorted({staging_store_token(s) for v in USER_STAGING_WRITES.values() for s in v}))
def test_every_function_touching_a_staged_store_is_classified(token: str, source: _Source) -> None:
    """A new reader of a staged store fails here until it is classified.

    Classifying a reader that serves other clients means naming the config keys
    that turn it on, and those keys are what the gate probes and what the
    all-readers-on template enables -- so neither can drift from the code.
    """
    assert token in STAGING_STORE_READERS, f"staged store {token!r} has no reader table"
    touching = source.touching(token, _terminals(token))
    unclassified = sorted(touching - set(STAGING_STORE_READERS[token]))
    assert unclassified == [], f"functions touching {token!r} with no reader classification: {unclassified}"
    stale = sorted(set(STAGING_STORE_READERS[token]) - set(source.funcs))
    assert stale == [], f"classified functions that no longer exist: {stale}"


def test_the_scan_is_not_blind(source: _Source) -> None:
    """Positive control: the validity gate reaches recall_outcome only through
    three hops of plumbing; the scan must find it, or it proves nothing."""
    touching = source.touching("recall_outcome", frozenset())
    assert "mind_mem.validity_gate._load_outcome_signals" in touching
    assert "mind_mem.outcome_attribution.report_outcome" in touching


def test_serving_readers_name_their_enabling_keys() -> None:
    """A serving reader that is always on makes its store un-stageable."""
    for token, readers in STAGING_STORE_READERS.items():
        for func, reader in readers.items():
            if reader.role == "serving":
                assert reader.enabled_by, f"{token}: {func} serves other clients unconditionally"


def test_approval_readers_are_admin_tools() -> None:
    """A reader that applies a staged row to shared state is an admin tool."""
    for token, readers in STAGING_STORE_READERS.items():
        for func, reader in readers.items():
            if reader.role == "approval" and func.startswith("mind_mem.mcp.tools."):
                assert func.rsplit(".", 1)[-1] in ADMIN_TOOLS, f"{token}: {func} applies staging but is not admin"


def test_reader_config_keys_cover_the_serving_readers() -> None:
    keys = set(cross_client_reader_config_keys())
    for readers in STAGING_STORE_READERS.values():
        for reader in readers.values():
            if reader.role == "serving":
                assert set(reader.enabled_by) <= keys


# ---------------------------------------------------------------------------
# The gate, end to end: report_outcome under the outcome-attribution reader
# ---------------------------------------------------------------------------

D1 = "D-20260101-001"
_OUTCOME = dict(block_ids=[D1], outcome="failure", task_id="t-reader")
_DECISION = f"\n[{D1}]\nDate: 2026-01-01\nStatus: active\nScope: global\nStatement: Use frost telemetry.\nTags: frost\n\n"
_OA_ON = {"enabled": True, "outcome_attribution": {"enabled": True}}


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from mind_mem.init_workspace import init

    root = tempfile.mkdtemp(prefix="mm_acl_readers_")
    ws = os.path.join(root, "ws")
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(_DECISION)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(ws, "mind-mem.json"))
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    try:
        yield ws
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _set_validity_gate(ws: str, section: dict[str, Any] | None) -> None:
    path = os.path.join(ws, "mind-mem.json")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall = cfg.setdefault("recall", {})
    if section is None:
        recall.pop("validity_gate", None)
    else:
        recall["validity_gate"] = section
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)


def _outcome_rows(ws: str) -> int:
    db = os.path.join(ws, ".mind-mem-index", "recall.db")
    if not os.path.exists(db):
        return 0
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT COUNT(*) FROM recall_outcome").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def _report(ws: str, scope: str, monkeypatch: pytest.MonkeyPatch) -> str:
    from mind_mem.mcp.tools.calibration import report_outcome

    monkeypatch.setenv("MIND_MEM_SCOPE", scope)
    from mind_mem.mcp.infra import rate_limit as _rate_limit

    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    with use_workspace(ws):
        return str(report_outcome(**_OUTCOME))


def test_outcome_attribution_reader_makes_report_outcome_admin(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_validity_gate(workspace, _OA_ON)
    result = _report(workspace, "user", monkeypatch)
    assert "requires admin scope" in result, result[:300]
    assert "outcome_attribution" in result, result[:300]
    assert _outcome_rows(workspace) == 0
    # Positive control: the identical call at admin scope lands the row.
    admin = json.loads(_report(workspace, "admin", monkeypatch))
    assert admin.get("status") == "recorded", admin
    assert _outcome_rows(workspace) == 1


@pytest.mark.parametrize(
    "section",
    [
        None,
        {"enabled": False, "outcome_attribution": {"enabled": True}},
        {"enabled": True, "outcome_attribution": {"enabled": False}},
        {"enabled": True},
    ],
    ids=["no-gate", "gate-off", "attribution-off", "attribution-absent"],
)
def test_report_outcome_stays_staging_while_the_reader_is_off(section: Any, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: with any one enabling key off the reader is dead, so the row is
    staging again -- the denial above is caused by the reader, nothing else."""
    _set_validity_gate(workspace, section)
    result = json.loads(_report(workspace, "user", monkeypatch))
    assert result.get("status") == "recorded", result
    assert _outcome_rows(workspace) == 1


def test_the_reader_the_gate_names_really_reads_the_store(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The enabling keys are the right ones: under them the validity gate
    actually loads outcome signals; with them off it does not."""
    from mind_mem.validity_gate import _load_outcome_signals

    assert _load_outcome_signals({"enabled": True}, workspace, [D1]) is None
    assert _load_outcome_signals(_OA_ON, workspace, [D1]) is not None


def test_malformed_config_fails_closed(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with open(os.path.join(workspace, "mind-mem.json"), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert check_write_mode_acl("report_outcome", dict(_OUTCOME, project_to_calibration=False), "user") is not None


def test_admin_scope_is_never_gated_by_a_reader(workspace: str) -> None:
    _set_validity_gate(workspace, _OA_ON)
    assert check_write_mode_acl("report_outcome", dict(_OUTCOME, project_to_calibration=False), "admin") is None
