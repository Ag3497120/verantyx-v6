"""
mcq_atom_cross_solver.py — Atom-based MCQ Cross Solver

cross_decomposeの進化版: キーワードoverlapではなくAtom構造マッチング

設計思想:
  1. 問題文(stem) → FactAtom化 → Wikipedia → FactAtom化
  2. 各選択肢 → FactAtom化 → Wikipedia → FactAtom化  
  3. stem_atoms × choice_atoms のAtom-level cross-matching
  4. supports/contradicts判定 → スコアリング
  5. LLM完全不使用

cross_decomposeとの違い:
  - キーワードoverlap → Atom supports/contradicts判定
  - 構造的マッチング → 精度向上
  - atom_relation_classifierを再利用
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AtomChoiceResult:
    label: str
    text: str
    concepts: List[str] = field(default_factory=list)
    wiki_texts: List[str] = field(default_factory=list)
    choice_atoms: list = field(default_factory=list)
    wiki_atoms: list = field(default_factory=list)
    supports: int = 0
    contradicts: int = 0
    weak_supports: int = 0
    score: float = 0.0


def solve_mcq_by_atom_cross(
    stem: str,
    choices: Dict[str, str],
    stem_facts: List[dict],
    ir_dict: Optional[dict] = None,
) -> Optional[Tuple[str, float, str]]:
    """
    Atom-based cross-matching MCQ solver.
    
    Returns: (answer_label, confidence, method) or None
    """
    if not choices or len(choices) < 2:
        return None

    # Import atomizer and classifier
    try:
        from knowledge.fact_atomizer import FactAtomizer
        _atomizer = FactAtomizer()
    except ImportError as e:
        log.warning(f"atom_cross: import error: {e}")
        return None

    # ── Step 1: stem facts → Atoms ──
    stem_texts = []
    for f in stem_facts:
        if isinstance(f, dict):
            s = f.get("summary", "") or f.get("plain", "")
            if s:
                stem_texts.append(s)
    
    stem_atoms = []
    for t in stem_texts:
        stem_atoms.extend(_atomizer.atomize(t))
    
    if not stem_atoms:
        log.debug("atom_cross: no stem atoms")
        return None

    # ── Step 2: 各選択肢を分解 → Wikipedia → Atom化 ──
    results = []
    for label, text in choices.items():
        cr = AtomChoiceResult(label=label, text=text)
        
        # 選択肢から概念抽出
        cr.concepts = _extract_concepts(text)
        
        # Wikipedia検索
        if cr.concepts:
            cr.wiki_texts = _fetch_wiki(cr.concepts)
        
        # 選択肢テキスト自体をAtom化
        cr.choice_atoms = list(_atomizer.atomize(text))
        
        # Wikipedia結果をAtom化
        for wt in cr.wiki_texts:
            cr.wiki_atoms.extend(_atomizer.atomize(wt))
        
        # ── Step 3: Cross-matching via atom pair comparison ──
        all_choice_atoms = cr.choice_atoms + cr.wiki_atoms
        
        if all_choice_atoms:
            for s_atom in stem_atoms:
                for c_atom in all_choice_atoms:
                    rel = _compare_atoms(s_atom, c_atom)
                    if rel == "supports":
                        cr.supports += 1
                    elif rel == "supports_weak":
                        cr.weak_supports += 1
                    elif rel == "contradicts":
                        cr.contradicts += 1
        
        # スコア計算: supports - contradicts (weak_supportsは0.3重み)
        cr.score = cr.supports + 0.3 * cr.weak_supports - 1.5 * cr.contradicts
        
        results.append(cr)
    
    # ── Step 4: ベスト選択 ──
    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]
    second = results[1] if len(results) > 1 else None
    
    gap = best.score - (second.score if second else 0)
    
    # 採用条件
    total_support = best.supports + best.weak_supports
    min_total_support = 2  # strong + weak合わせて最低2
    min_gap = 0.5          # スコア差
    
    if total_support >= min_total_support and gap >= min_gap and best.score > 0:
        confidence = min(0.70, 0.30 + gap * 0.05 + best.supports * 0.03)
        method = (
            f"atom_cross:best={best.label}"
            f"(score={best.score:.1f},gap={gap:.1f}"
            f",sup={best.supports},weak={best.weak_supports}"
            f",contra={best.contradicts}"
            f",atoms={len(best.choice_atoms)+len(best.wiki_atoms)}"
            f",stem_atoms={len(stem_atoms)})"
        )
        log.info(f"atom_cross: {method}")
        return best.label, confidence, method
    
    log.debug(
        f"atom_cross: no_winner best={best.label}(score={best.score:.1f},"
        f"sup={best.supports},gap={gap:.1f})"
    )
    return None


# ── 概念抽出 (cross_decomposeから流用+強化) ──

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'and', 'or', 'but', 'for',
    'with', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'it', 'its',
    'this', 'that', 'which', 'who', 'whom', 'whose', 'what', 'where',
    'when', 'how', 'if', 'then', 'than', 'not', 'no', 'only', 'also',
}


def _extract_concepts(text: str) -> List[str]:
    """選択肢テキストから概念を抽出"""
    concepts = []
    
    # 固有名詞パターン
    for m in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
        term = m.group(1).strip()
        if len(term) > 3 and term.lower() not in _STOPWORDS:
            concepts.append(term)
    
    # ハイフン付き用語
    for m in re.finditer(r'([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)', text):
        term = m.group(1).strip()
        if len(term) > 5:
            concepts.append(term)
    
    # 括弧付き説明
    for m in re.finditer(r'\(([^)]+)\)', text):
        inner = m.group(1).strip()
        if 3 < len(inner) < 50:
            concepts.append(inner)
    
    # テキスト全体が短い場合はそのまま
    if not concepts and len(text.split()) <= 6:
        clean = re.sub(r'[^\w\s-]', '', text).strip()
        if clean and clean.lower() not in _STOPWORDS:
            concepts.append(clean)
    
    return concepts[:5]  # 最大5概念


_ANTONYMS = {
    'increase': 'decrease', 'high': 'low', 'large': 'small', 'true': 'false',
    'positive': 'negative', 'active': 'inactive', 'present': 'absent',
    'common': 'rare', 'major': 'minor', 'strong': 'weak', 'open': 'closed',
    'north': 'south', 'east': 'west', 'up': 'down', 'left': 'right',
    'male': 'female', 'internal': 'external', 'above': 'below',
}
# Bidirectional
_ANTONYM_MAP = {}
for k, v in _ANTONYMS.items():
    _ANTONYM_MAP[k] = v
    _ANTONYM_MAP[v] = k


def _compare_atoms(atom_a, atom_b) -> str:
    """Compare two FactAtoms: supports / supports_weak / contradicts / unknown"""
    s_a = (getattr(atom_a, 'subject', '') or '').lower().strip()
    p_a = (getattr(atom_a, 'predicate', '') or '').lower().strip()
    o_a = (getattr(atom_a, 'object', '') or '').lower().strip()
    s_b = (getattr(atom_b, 'subject', '') or '').lower().strip()
    p_b = (getattr(atom_b, 'predicate', '') or '').lower().strip()
    o_b = (getattr(atom_b, 'object', '') or '').lower().strip()

    if not (s_a and s_b):
        return "unknown"

    # Subject overlap check
    subj_match = (
        s_a == s_b or s_a in s_b or s_b in s_a
        or bool(set(s_a.split()) & set(s_b.split()) - _STOPWORDS)
    )
    if not subj_match:
        # Check if object of one matches subject of other (transitive)
        if o_a and (o_a == s_b or o_a in s_b or s_b in o_a):
            subj_match = True
        elif o_b and (o_b == s_a or o_b in s_a or s_a in o_b):
            subj_match = True

    if not subj_match:
        return "unknown"

    # Same predicate + same object → strong support
    pred_match = p_a == p_b or p_a in p_b or p_b in p_a
    obj_match = o_a and o_b and (o_a == o_b or o_a in o_b or o_b in o_a)

    if pred_match and obj_match:
        return "supports"

    # Same predicate, different object → might contradict
    if pred_match and o_a and o_b and not obj_match:
        # Check antonyms
        if o_a in _ANTONYM_MAP and _ANTONYM_MAP[o_a] == o_b:
            return "contradicts"
        # Numeric contradiction
        try:
            na, nb = float(o_a), float(o_b)
            if na != nb:
                return "contradicts"
        except (ValueError, TypeError):
            pass

    # Keyword overlap in object/predicate → weak support
    all_a = set(f"{p_a} {o_a}".split()) - _STOPWORDS
    all_b = set(f"{p_b} {o_b}".split()) - _STOPWORDS
    overlap = all_a & all_b
    if len(overlap) >= 2:
        return "supports_weak"
    if len(overlap) >= 1 and (pred_match or obj_match):
        return "supports_weak"

    # Negation check
    neg_a = 'not' in p_a or 'never' in p_a or p_a.startswith('un') or p_a.startswith('non')
    neg_b = 'not' in p_b or 'never' in p_b or p_b.startswith('un') or p_b.startswith('non')
    if neg_a != neg_b and (pred_match or obj_match):
        return "contradicts"

    return "unknown"


def _fetch_wiki(concepts: List[str]) -> List[str]:
    """概念リストからWikipedia要約を取得 (cross_decomposeと同じfetcher)"""
    texts = []
    try:
        from knowledge.wiki_knowledge_fetcher_v2 import WikiKnowledgeFetcherV2
        fetcher = WikiKnowledgeFetcherV2()
        for concept in concepts[:3]:
            try:
                result = fetcher.fetch(concept)
                if result and result.found and result.facts:
                    for wf in result.facts[:2]:
                        summary = (wf.summary if hasattr(wf, 'summary') else str(wf))[:500]
                        if summary:
                            texts.append(summary)
                elif result and result.raw_text:
                    texts.append(result.raw_text[:500])
            except Exception:
                continue
    except Exception:
        pass
    return texts
