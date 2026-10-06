"""Session expansion: answer from whole conversation sessions instead of isolated turns.

A recalled turn is rarely self-sufficient in conversational memory: the fact it points at is often
stated one or two turns away (the question in one turn, the answer in the next), a list is spread
over a session, and a date is only meaningful next to the session it belongs to. Session expansion
takes the recall hits, ranks the SESSIONS they belong to (best hit first, then every other session
by lexical similarity of the whole session to the question), and returns whole sessions, in
chronological order, until a character budget is spent.

Pure functions over an admitted corpus (``{block_id: block}``); callers pass a corpus that already
went through the admission gate, so nothing withheld can be pulled in by expansion.

Session identity of a block, in order: an explicit ``Session`` field, a ``session-<n>`` tag, else
the block's ``Date``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

__all__ = ["session_of", "rank_sessions", "expand_sessions", "render_transcript"]

_WORD = re.compile(r"[a-z0-9]+")
_SESSION_TAG = re.compile(r"\bsession-(\d+)\b", re.IGNORECASE)
_STOP = frozenset(
    "the a an of to in on at for and or is was were did do does i my me you your what when where who which how "
    "with about from that this it be have has had he she they them his her their its by as".split()
)


def _terms(text: str) -> list[str]:
    """Lower-cased word prefixes (6 chars) without stop words: a cheap, dependency-free stemmer."""
    return [w[:6] for w in _WORD.findall(text.lower()) if w not in _STOP]


def _text(block: Mapping[str, Any]) -> str:
    for key in ("Statement", "excerpt", "Title", "Summary", "Description"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def session_of(block: Mapping[str, Any]) -> str:
    """The session key of a block ('' when it has none)."""
    sess = block.get("Session")
    if isinstance(sess, (str, int)) and str(sess).strip():
        return f"S:{str(sess).strip()}"
    tags = block.get("Tags") or block.get("tags") or ""
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(str(t) for t in tags)
    m = _SESSION_TAG.search(str(tags))
    if m:
        return f"T:{int(m.group(1))}"
    date = block.get("Date") or block.get("date")
    return f"D:{date}" if isinstance(date, str) and date.strip() else ""


def _group(corpus: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for block_id, block in corpus.items():
        key = session_of(block)
        if key:
            groups.setdefault(key, []).append(block_id)
    return groups


def _bm25(question: str, docs: Mapping[str, list[str]], k1: float = 1.2, b: float = 0.75) -> dict[str, float]:
    n = len(docs)
    if not n:
        return {}
    avg = sum(len(d) for d in docs.values()) / n or 1.0
    df: dict[str, int] = {}
    for d in docs.values():
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    query = set(_terms(question))
    out: dict[str, float] = {}
    for key, d in docs.items():
        tf: dict[str, int] = {}
        for w in d:
            if w in query:
                tf[w] = tf.get(w, 0) + 1
        out[key] = sum(
            math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
            for w, f in tf.items()
        )
    return out


def rank_sessions(question: str, hits: Sequence[Mapping[str, Any]], corpus: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Session keys: sessions of the hits in hit order, then all others by whole-session BM25."""
    groups = _group(corpus)
    ranked: list[str] = []
    for hit in hits:
        block_id = str(hit.get("_id", "")).split("::")[0]
        block = corpus.get(block_id)
        key = session_of(block) if block is not None else ""
        if key and key not in ranked:
            ranked.append(key)
    docs = {key: _terms(" ".join(_text(corpus[b]) for b in ids)) for key, ids in groups.items() if key not in ranked}
    lex = _bm25(question, docs)
    ranked += sorted(lex, key=lambda k: (-lex[k], k))
    return ranked


def expand_sessions(
    question: str,
    hits: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]],
    budget_chars: int,
) -> list[dict[str, Any]]:
    """Whole sessions around the hits, within *budget_chars* of block text.

    Sessions are taken in :func:`rank_sessions` order; a session that does not fit the remaining
    budget is skipped (a later, smaller one may still fit). The result lists the chosen sessions in
    corpus order (chronological for an append-only store) and each session's blocks in corpus
    order. Each item is a hit-shaped dict: ``_id``, full text as ``excerpt``, and the block's
    ``Date`` / ``speaker`` / ``DiaID`` / source fields when present. Returns ``[]`` for a
    non-positive budget, so a disabled expansion cannot change the caller's evidence.
    """
    if budget_chars <= 0 or not corpus:
        return []
    groups = _group(corpus)
    chosen: set[str] = set()
    used = 0
    for key in rank_sessions(question, hits, corpus):
        cost = sum(len(_text(corpus[b])) + 1 for b in groups.get(key, ()))
        if cost and used + cost <= budget_chars:
            chosen.add(key)
            used += cost
    out: list[dict[str, Any]] = []
    for block_id, block in corpus.items():
        if session_of(block) not in chosen:
            continue
        item: dict[str, Any] = {"_id": block_id, "excerpt": _text(block), "score": 0.0, "session": session_of(block)}
        for key in ("Date", "speaker", "DiaID", "_source_file", "file", "Status"):
            if block.get(key):
                item[key] = block[key]
        out.append(item)
    return out


def render_transcript(items: Sequence[Mapping[str, Any]]) -> str:
    """Compact transcript: one header per session (with its date), then ``<source> <text>`` lines."""
    lines: list[str] = []
    current = None
    for item in items:
        key = item.get("session")
        if key != current:
            current = key
            date = item.get("Date") or ""
            lines.append(f"--- session {str(key).split(':', 1)[-1]}{' (' + date + ')' if date else ''} ---")
        source = item.get("DiaID") or item.get("_id")
        lines.append(f"{source} {item.get('excerpt', '')}")
    return "\n".join(lines)
