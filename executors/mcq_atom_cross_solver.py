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
        from knowledge.fact_atomizer import atomize_many
        from executors.atom_relation_classifier import classify_relations_by_atoms
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
        stem_atoms.extend(atomize_many(t))
    
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
        cr.choice_atoms = list(atomize_many(text))
        
        # Wikipedia結果をAtom化
        for wt in cr.wiki_texts:
            cr.wiki_atoms.extend(atomize_many(wt))
        
        # ── Step 3: Cross-matching via atom classifier ──
        # stem_atoms vs choice_wiki_atoms (stemの知識が選択肢のWiki事実を支持するか)
        all_choice_atoms = cr.choice_atoms + cr.wiki_atoms
        
        if all_choice_atoms:
            for s_atom in stem_atoms:
                for c_atom in all_choice_atoms:
                    rel = classify_relations_by_atoms(
                        [s_atom], [c_atom]
                    )
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
    min_supports = 2  # 最低2つのsupports
    min_gap = 1.0     # 最低1.0のスコア差
    
    if best.supports >= min_supports and gap >= min_gap and best.score > 0:
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


def _fetch_wiki(concepts: List[str]) -> List[str]:
    """概念リストからWikipedia要約を取得"""
    texts = []
    try:
        from knowledge.knowledge_pipeline_v2 import fetch_wikipedia_summary
    except ImportError:
        try:
            import wikipedia
            def fetch_wikipedia_summary(query):
                try:
                    page = wikipedia.page(query, auto_suggest=True)
                    return page.summary[:500]
                except:
                    return None
        except ImportError:
            return texts
    
    for concept in concepts[:3]:  # 最大3概念
        try:
            summary = fetch_wikipedia_summary(concept)
            if summary:
                texts.append(summary if isinstance(summary, str) else str(summary))
        except Exception:
            continue
    
    return texts
