# Copyright 2026 STARGA, Inc.
"""Give the user back exactly what they had when mind-mem is uninstalled.

Removing the ``mind-mem`` entry from a client config re-serialises the file,
which changes its formatting (JSON indent, key spacing) even though no user
content changed. Two cases are restored exactly instead:

* the file did not exist before install: what is left after removal is only
  the empty skeleton mind-mem itself created (``{"mcpServers": {}}``, an
  empty ``extensions:`` map ...), so the file is deleted;
* the file existed: install took a ``<file>.bak-mind-mem-*`` backup first.
  When a backup holds the same document as the post-removal content, its
  bytes are written back, so formatting and comments return unchanged.

Both checks compare parsed documents, never guess: a user edit made after
install (another server, a changed setting) makes the post-removal document
differ from the skeleton and from every backup, and the plain re-serialised
removal is kept. Nothing here deletes anything a user wrote.
"""

from __future__ import annotations

import glob
import json
from typing import Any

from mind_mem.client_config_io import BACKUP_INFIX
from mind_mem.client_writers import JSON_FORMATS, JSON_WRITERS, TEXT_FORMATS, remove_json_mcp, remove_opencode_mcp

_PROBE_SRV: dict[str, Any] = {"command": "mind-mem-mcp", "args": [], "env": {}}
_PROBE_WS = "/nonexistent/mind-mem-workspace"


def _kind(fmt: str) -> str:
    if fmt in JSON_FORMATS or fmt == "mcp-json-opencode":
        return "json"
    if fmt.startswith("mcp-toml"):
        return "toml"
    return "yaml"


def _parse(kind: str, text: str) -> Any:
    """Parse *text*; raises on anything that is not a clean document."""
    if kind == "json":
        return json.loads(text)
    if kind == "toml":
        try:
            import tomllib as toml  # noqa: PLC0415
        except ImportError:  # pragma: no cover - Python 3.10
            import tomli as toml  # type: ignore[no-redef]  # noqa: PLC0415
        return toml.loads(text)
    import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

    return yaml.safe_load(text)


def _same(kind: str, a: str, b: str) -> bool:
    """True when *a* and *b* hold the same document (exact text otherwise)."""
    if a == b:
        return True
    try:
        return bool(_parse(kind, a) == _parse(kind, b))
    except Exception:  # noqa: BLE001 - unparseable or parser missing: exact text only
        return a.rstrip("\n") == b.rstrip("\n")


def skeleton(fmt: str) -> str:
    """What remains after removing mind-mem from a file mind-mem created from nothing."""
    if fmt == "mcp-json-opencode":
        from mind_mem.opencode_config import merge_opencode_mcp  # noqa: PLC0415

        return json.dumps(remove_opencode_mcp(merge_opencode_mcp({}, _PROBE_SRV)[0])[0], indent=2)
    if fmt in JSON_WRITERS:
        created = JSON_WRITERS[fmt]({}, _PROBE_WS, _PROBE_SRV)[0]
        return json.dumps(remove_json_mcp(JSON_FORMATS[fmt][0], created)[0], indent=2)
    write, remove = TEXT_FORMATS[fmt]
    return remove(write("", _PROBE_WS, _PROBE_SRV)[0])[0]


def is_skeleton(fmt: str, removed_text: str) -> bool:
    """The post-removal content is only what mind-mem itself created."""
    kind = _kind(fmt)
    try:
        return bool(_parse(kind, removed_text) == _parse(kind, skeleton(fmt))) or not removed_text.strip()
    except Exception:  # noqa: BLE001
        return removed_text.strip() == skeleton(fmt).strip()


def matching_backup(fmt: str, path: str, removed_text: str) -> str | None:
    """Text of the newest ``<path>.bak-mind-mem-*`` equal to *removed_text*, if any."""
    kind = _kind(fmt)
    candidates = sorted(glob.glob(glob.escape(path) + BACKUP_INFIX + "*"), reverse=True)  # newest stamp first
    for backup in candidates:
        try:
            with open(backup, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if _same(kind, text, removed_text):
            return text
    return None


__all__ = ["is_skeleton", "matching_backup", "skeleton"]
