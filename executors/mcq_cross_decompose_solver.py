"""
mcq_cross_decompose_solver.py
選択肢分解 × Wikipedia cross-matching MCQ ソルバー

設計思想（kofdai 2026-02-22 21:47 提案）:
  1. 選択肢ラベル（A/B/C/1/2/3/あ/い/う）を動的検出
  2. 各選択肢を個別に分解（概念抽出）
  3. stem と各選択肢をそれぞれ Wikipedia に投げる
  4. stem_facts × choice_facts の cross-matching でスコアリング
  5. LLM不使用 → position bias ゼロ、完全ルールベース

利点:
  - LLMに選択肢を一括で渡さない → position bias 完全排除
  - 各選択肢が固有の Wikipedia facts を取得 → 精度向上
  - cross-matching はルールベース → 再現性あり

鉄の壁準拠: 問題文もLLMに渡さない。Wikipedia API のみ使用。
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── 選択肢ラベル検出パターン ──
LABEL_PATTERNS = [
    # A, B, C, D, E ...
    re.compile(r'^([A-Z])[\.\):\s]'),
    # a, b, c, d, e ...
    re.compile(r'^([a-z])[\.\):\s]'),
    # 1, 2, 3, 4, 5 ...
    re.compile(r'^(\d+)[\.\):\s]'),
    # あ, い, う, え, お
    re.compile(r'^([あいうえおかきくけこ])[\.\):\s]'),
    # ア, イ, ウ, エ, オ
    re.compile(r'^([アイウエオカキクケコ])[\.\):\s]'),
    # (A), (B), (C) ...
    re.compile(r'^\(([A-Za-z\d])\)'),
    # ①, ②, ③ ...
    re.compile(r'^([①②③④⑤⑥⑦⑧⑨⑩])'),
]

# 分解価値がある選択肢の最小文字数
MIN_CHOICE_LEN_FOR_DECOMPOSE = 15

# ── ストップワード ──
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'and', 'or', 'but', 'for',
    'with', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'it', 'its',
    'this', 'that', 'which', 'who', 'whom', 'whose', 'what', 'where',
    'when', 'how', 'if', 'then', 'than', 'both', 'each', 'all', 'any',
    'not', 'no', 'nor', 'only', 'also', 'such', 'so', 'too', 'very',
    'just', 'about', 'more', 'most', 'other', 'some', 'many', 'much',
}


@dataclass
class ChoiceDecomposition:
    """選択肢の分解結果"""
    label: str
    text: str
    concepts: List[str] = field(default_factory=list)
    wiki_facts: List[str] = field(default_factory=list)
    cross_score: float = 0.0
    overlap_terms: List[str] = field(default_factory=list)


@dataclass
class CrossMatchResult:
    """Cross-matching 結果"""
    answer: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    decompositions: List[ChoiceDecomposition] = field(default_factory=list)
    stem_concepts: List[str] = field(default_factory=list)
    stem_facts_count: int = 0
    reject_reason: str = ""


def solve_by_cross_decomposition(
    stem: str,
    choices: Dict[str, str],
    stem_facts: List[dict],
    ir_dict: Optional[dict] = None,
) -> Optional[Tuple[str, float, str]]:
    """
    選択肢分解 × cross-matching で MCQ を解く。

    Args:
        stem: 問題文のstem部分
        choices: {"A": "text", "B": "text", ...}
        stem_facts: pipeline が既に取得済みの stem 用 Wikipedia facts
        ir_dict: IR の to_dict() 出力（概念抽出用）

    Returns:
        (answer_label, confidence, method) or None
    """
    if not choices or len(choices) < 2:
        return None

    # 短い選択肢はスキップ（数値/記号のみ）
    avg_len = sum(len(v) for v in choices.values()) / len(choices)
    if avg_len < MIN_CHOICE_LEN_FOR_DECOMPOSE:
        log.debug(f"cross_decompose: skip, avg choice len={avg_len:.0f} < {MIN_CHOICE_LEN_FOR_DECOMPOSE}")
        return None

    # ── Step 1: stem facts からキーワード集合を構築 ──
    stem_keywords = _extract_keywords_from_facts(stem_facts)

    # IR の entities/missing からもキーワード追加
    if ir_dict:
        for e in ir_dict.get("entities", []):
            name = e.get("name", "") if isinstance(e, dict) else str(e)
            if name:
                stem_keywords.update(_tokenize(name))
        for m in ir_dict.get("missing", []):
            concept = m.get("concept", "") if isinstance(m, dict) else str(m)
            if concept:
                stem_keywords.update(_tokenize(concept.replace("_", " ")))

    # ── Step 2: 各選択肢を分解してWikipedia検索 ──
    decompositions = []
    for label, text in choices.items():
        cd = ChoiceDecomposition(label=label, text=text)

        # 選択肢から概念を抽出
        cd.concepts = _extract_concepts_from_choice(text)

        # Wikipedia 検索（各選択肢固有）
        if cd.concepts:
            cd.wiki_facts = _fetch_wikipedia_for_concepts(cd.concepts)

        decompositions.append(cd)

    # ── Step 3: Cross-matching (stem_facts × choice_facts) ──
    for cd in decompositions:
        choice_keywords = _tokenize(cd.text)
        choice_fact_keywords = set()
        for fact in cd.wiki_facts:
            choice_fact_keywords.update(_tokenize(fact))

        # Score 1: stem_facts のキーワードが choice の Wikipedia facts に出現する割合
        if stem_keywords and choice_fact_keywords:
            overlap_stem_in_choice = stem_keywords & choice_fact_keywords
            score1 = len(overlap_stem_in_choice) / max(len(stem_keywords), 1)
        else:
            score1 = 0.0

        # Score 2: choice のキーワードが stem_facts に出現する割合
        stem_fact_keywords = set()
        for f in stem_facts:
            if isinstance(f, dict):
                s = f.get("summary", "") or f.get("plain", "")
                stem_fact_keywords.update(_tokenize(s))
        if choice_keywords and stem_fact_keywords:
            overlap_choice_in_stem = choice_keywords & stem_fact_keywords
            score2 = len(overlap_choice_in_stem) / max(len(choice_keywords), 1)
        else:
            score2 = 0.0

        # Score 3: choice の Wikipedia facts と choice text の一致度（自己確認）
        if choice_keywords and choice_fact_keywords:
            self_overlap = choice_keywords & choice_fact_keywords
            score3 = len(self_overlap) / max(len(choice_keywords), 1)
        else:
            score3 = 0.0

        # 総合スコア（重み付き）
        cd.cross_score = 0.4 * score1 + 0.3 * score2 + 0.3 * score3

        # デバッグ用 overlap terms
        all_overlaps = set()
        if stem_keywords and choice_fact_keywords:
            all_overlaps.update(stem_keywords & choice_fact_keywords)
        if choice_keywords and stem_fact_keywords:
            all_overlaps.update(choice_keywords & stem_fact_keywords)
        cd.overlap_terms = sorted(list(all_overlaps))[:10]

    # ── Step 4: 最高スコアの選択肢を選択 ──
    decompositions.sort(key=lambda d: d.cross_score, reverse=True)
    best = decompositions[0]
    second = decompositions[1] if len(decompositions) > 1 else None

    # 差分が十分大きい場合のみ回答
    gap = best.cross_score - (second.cross_score if second else 0)
    min_score = 0.05  # 最低スコア閾値
    # 選択肢数に応じた動的gap閾値（多い選択肢ほど高いgapを要求）
    n_choices = len(decompositions)
    if n_choices <= 4:
        min_gap = 0.03
    elif n_choices <= 6:
        min_gap = 0.045
    else:
        min_gap = 0.06  # 7択以上: ランダムノイズと区別するため厳しく

    result = CrossMatchResult(
        decompositions=decompositions,
        stem_concepts=list(stem_keywords)[:20],
        stem_facts_count=len(stem_facts),
    )

    if best.cross_score >= min_score and gap >= min_gap:
        result.answer = best.label
        result.confidence = min(0.65, 0.35 + gap * 3 + best.cross_score)
        result.method = (
            f"cross_decompose:best={best.label}"
            f"(score={best.cross_score:.3f},gap={gap:.3f}"
            f",concepts={len(best.concepts)}"
            f",wiki_facts={len(best.wiki_facts)}"
            f",overlaps={len(best.overlap_terms)})"
        )
        log.info(f"cross_decompose: {result.method}")
        return result.answer, result.confidence, result.method

    result.reject_reason = (
        f"no_clear_winner(best={best.label}:{best.cross_score:.3f}"
        f",gap={gap:.3f},min_score={min_score},min_gap={min_gap})"
    )
    log.debug(f"cross_decompose: {result.reject_reason}")
    return None


def _extract_concepts_from_choice(text: str) -> List[str]:
    """選択肢テキストから概念（Wikipedia検索用クエリ）を抽出"""
    concepts = []

    # 大文字で始まる複合語（固有名詞、専門用語）
    for m in re.finditer(r'(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
        term = m.group(1).strip()
        if len(term) > 3 and term.lower() not in STOPWORDS:
            concepts.append(term)

    # ハイフン付き用語
    for m in re.finditer(r'([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)', text):
        term = m.group(1).strip()
        if len(term) > 5:
            concepts.append(term)

    # 所有格パターン（X's Y）
    for m in re.finditer(r"([A-Z][a-z]+(?:'s)?\s+[a-z]+(?:\s+[a-z]+)?)", text):
        term = m.group(1).strip()
        if len(term) > 5 and term.lower() not in STOPWORDS:
            concepts.append(term)

    # 括弧内の用語
    for m in re.finditer(r'\(([^)]{3,40})\)', text):
        inner = m.group(1).strip()
        if not re.match(r'^[\d\s,.\-]+$', inner):
            concepts.append(inner)

    # 重複除去（順序保持）
    seen = set()
    unique = []
    for c in concepts:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique[:5]  # 最大5概念


def _fetch_wikipedia_for_concepts(concepts: List[str]) -> List[str]:
    """概念リストをWikipediaで検索してファクトを取得"""
    facts = []
    try:
        from knowledge.wiki_knowledge_fetcher_v2 import WikiKnowledgeFetcherV2
        fetcher = WikiKnowledgeFetcherV2()

        for concept in concepts[:2]:  # 最大2概念（速度のため、選択肢×5 = 10 API呼び出し上限）
            try:
                result = fetcher.fetch(concept)
                if result and result.found and result.facts:
                    for wf in result.facts[:2]:
                        summary = (wf.summary if hasattr(wf, 'summary') else str(wf))[:300]
                        if summary:
                            facts.append(summary)
                elif result and result.raw_text:
                    facts.append(result.raw_text[:300])
            except Exception as e:
                log.debug(f"wiki fetch for '{concept}': {e}")
                continue
    except Exception as e:
        log.debug(f"wiki fetcher init error: {e}")

    return facts


def _extract_keywords_from_facts(facts: List[dict]) -> set:
    """facts リストからキーワード集合を抽出"""
    keywords = set()
    for f in facts:
        if isinstance(f, dict):
            text = f.get("summary", "") or f.get("plain", "")
            keywords.update(_tokenize(text))
            for p in f.get("properties", []):
                keywords.update(_tokenize(str(p)))
        elif isinstance(f, str):
            keywords.update(_tokenize(f))
    return keywords


def _tokenize(text: str) -> set:
    """テキストをトークン化（ストップワード除去、3文字以上）"""
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return {w for w in words if w not in STOPWORDS}
