"""
KB Loader — loads foundation_law_kb.jsonl and exposes a search interface.

Integrated into the pipeline via executors/knowledge_lookup.py.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

_KB_PATH = Path(__file__).parent / "foundation_law_kb.jsonl"
_entries: List[Dict[str, Any]] = []
_loaded = False


def _load():
    global _entries, _loaded
    if _loaded:
        return
    if not _KB_PATH.exists():
        _loaded = True
        return
    with open(_KB_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    _entries.append(json.loads(line))
                except Exception:
                    pass
    _loaded = True


def search_kb(
    question: str,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for entries relevant to `question`.
    Returns up to `top_k` entries sorted by relevance.
    """
    _load()
    if not _entries:
        return []

    q_lower = question.lower()
    scored = []

    for entry in _entries:
        score = 0.0

        # Domain filter (loose match)
        if domain:
            entry_domain = entry.get("domain", "")
            if domain.lower() in entry_domain.lower() or entry_domain.lower() in domain.lower():
                score += 2.0
            # Check tags too
            for tag in entry.get("tags", []):
                if domain.lower() in tag.lower():
                    score += 1.0

        # Keyword matching
        for kw in entry.get("keywords", []):
            kw_l = kw.lower()
            if kw_l in q_lower:
                # Exact keyword hit
                score += 2.0
                # Bonus for multi-word matches
                if " " in kw_l:
                    score += 1.0

        # Statement keyword overlap
        stmt = entry.get("statement", "").lower()
        stmt_words = set(re.findall(r'\b[a-z]{4,}\b', stmt))
        q_words = set(re.findall(r'\b[a-z]{4,}\b', q_lower))
        overlap = stmt_words & q_words
        score += 0.3 * len(overlap)

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def get_known_value(entry_id: str, key: str) -> Optional[Any]:
    """Retrieve a specific known value from an entry."""
    _load()
    for entry in _entries:
        if entry.get("id") == entry_id:
            kv = entry.get("known_values", {})
            return kv.get(key)
    return None


def get_entry_by_id(entry_id: str) -> Optional[Dict[str, Any]]:
    """Look up an entry by its id."""
    _load()
    for entry in _entries:
        if entry.get("id") == entry_id:
            return entry
    return None


def get_formula(entry: Dict[str, Any]) -> Optional[str]:
    """Extract formula string from an entry if present."""
    return entry.get("formula") or entry.get("statement")


def count() -> int:
    _load()
    return len(_entries)


# ── Direct known-value lookups (highest-confidence) ──

def lookup_sequence_value(seq_name: str, n: int) -> Optional[int]:
    """
    Look up the n-th value in a named sequence.
    seq_name: 'catalan', 'bell', 'fibonacci', 'lucas', 'partition', etc.
    """
    _load()
    seq_map = {
        "catalan":   ("comb_001", f"C_{n}"),
        "bell":      ("comb_002", f"B_{n}"),
        "fibonacci": ("comb_006", f"F_{n}"),
        "lucas":     ("comb_007", f"L_{n}"),
        "partition": ("comb_008", f"p({n})"),
        "derangement": ("comb_004", f"D_{n}"),
        "motzkin":   ("comb_011", f"M_{n}"),
    }
    if seq_name.lower() not in seq_map:
        return None
    entry_id, key = seq_map[seq_name.lower()]
    return get_known_value(entry_id, key)


def lookup_ramsey(s: int, t: int) -> Optional[int]:
    """Look up Ramsey number R(s,t)."""
    _load()
    entry = get_entry_by_id("comb_005")
    if entry:
        kv = entry.get("known_values", {})
        return kv.get(f"R({s},{t})")
    return None


def lookup_group_order(group_name: str, n: int) -> Optional[int]:
    """Look up the order of common groups."""
    name_lower = group_name.lower()
    if "symmetric" in name_lower or name_lower.startswith("s_"):
        entry = get_entry_by_id("group_003")
        if entry:
            return entry.get("known_values", {}).get(f"S_{n}")
    if "dihedral" in name_lower or name_lower.startswith("d_"):
        return 2 * n
    if "alternating" in name_lower or name_lower.startswith("a_"):
        entry = get_entry_by_id("group_007")
        if entry:
            return entry.get("known_values", {}).get(f"A_{n}")
    return None


if __name__ == "__main__":
    print(f"KB loaded: {count()} entries")
    results = search_kb("What is the 7th Fibonacci number?", domain="combinatorics")
    for r in results:
        print(f"  {r['id']}: {r['statement'][:80]}")
    print(f"\nFib(10) = {lookup_sequence_value('fibonacci', 10)}")
    print(f"R(3,3) = {lookup_ramsey(3,3)}")
    print(f"|S_5| = {lookup_group_order('symmetric', 5)}")
