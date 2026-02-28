"""
mcq_cross_elimination.py
========================
ARC cross構造の消去法をHLE MCQに移植。

設計思想（ARCからの移植）:
  ARC: 各セルの近傍パターンから規則を学習 → テストに適用
  HLE: 問題文の制約キーワードで各選択肢を検証 → 矛盾する選択肢を消去

Cross軸:
  +X: 問題文から抽出した制約/条件
  -X: 各選択肢の主張
  +Y: 制約→選択肢のマッチング（支持）
  -Y: 制約→選択肢の矛盾（消去）
  +Z: 選択肢間の相互排他性チェック
  -Z: 消去ログ

LLMフリー: 全てルールベース。Wikipedia不使用。
"""

from __future__ import annotations
import re
import math
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Constraint:
    """問題文から抽出した制約"""
    text: str
    keywords: Set[str]
    negated: bool = False  # "not", "cannot", "never" etc
    numeric_value: Optional[float] = None
    comparison: Optional[str] = None  # >, <, =, >=, <=


@dataclass 
class ChoiceCross:
    """選択肢のcross構造"""
    label: str
    text: str
    keywords: Set[str]
    support_score: float = 0.0
    contradiction_score: float = 0.0
    eliminated: bool = False
    elimination_reason: str = ""


NEGATION_WORDS = {'not', 'never', 'no', 'cannot', 'neither', 'nor', 'without', 
                  'impossible', 'incorrect', 'false', 'wrong', 'invalid',
                  'doesn\'t', 'don\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
                  'won\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t'}

QUANTITY_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(percent|%|times|fold|orders?\s+of\s+magnitude|nm|um|mm|cm|m|kg|g|mg|μg|mol|molar|hz|khz|mhz|ghz|ev|kev|mev|gev|tev|joules?|watts?|volts?|amps?|ohms?|seconds?|minutes?|hours?|days?|years?|bits?|bytes?|kb|mb|gb|tb)', re.I)

COMPARATIVE_WORDS = {
    'greater': '>', 'larger': '>', 'higher': '>', 'more': '>', 'above': '>',
    'increases': '>', 'increase': '>', 'increased': '>',
    'less': '<', 'smaller': '<', 'lower': '<', 'fewer': '<', 'below': '<',
    'decreases': '<', 'decrease': '<', 'decreased': '<',
    'equal': '=', 'same': '=', 'identical': '=', 'equivalent': '=',
    'at least': '>=', 'at most': '<=', 'maximum': '<=', 'minimum': '>=',
}

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'and', 'or', 'but', 'for',
    'with', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'it', 'its',
    'this', 'that', 'which', 'who', 'what', 'where', 'when', 'how',
    'if', 'then', 'than', 'both', 'each', 'all', 'any', 'some',
}


def _tokenize(text: str) -> Set[str]:
    """テキストをキーワードセットに変換"""
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return set(words) - STOPWORDS


def _extract_constraints(problem_text: str) -> List[Constraint]:
    """問題文から制約を抽出"""
    constraints = []
    
    # 文に分割
    sentences = re.split(r'[.!?]\s+', problem_text)
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        keywords = _tokenize(sent)
        if len(keywords) < 2:
            continue
        
        # 否定チェック
        sent_lower = sent.lower()
        negated = any(neg in sent_lower.split() for neg in NEGATION_WORDS)
        
        # 数値抽出
        num_match = QUANTITY_PATTERN.search(sent)
        numeric_value = float(num_match.group(1)) if num_match else None
        
        # 比較語チェック
        comparison = None
        for word, comp in COMPARATIVE_WORDS.items():
            if word in sent_lower:
                comparison = comp
                break
        
        constraints.append(Constraint(
            text=sent,
            keywords=keywords,
            negated=negated,
            numeric_value=numeric_value,
            comparison=comparison,
        ))
    
    return constraints


def _build_choice_cross(label: str, text: str) -> ChoiceCross:
    """選択肢のcross構造を構築"""
    return ChoiceCross(
        label=label,
        text=text,
        keywords=_tokenize(text),
    )


def _compute_keyword_overlap(set_a: Set[str], set_b: Set[str]) -> float:
    """2つのキーワードセットの重複率（Jaccard-like）"""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _check_numeric_contradiction(constraint: Constraint, choice: ChoiceCross) -> bool:
    """数値的矛盾をチェック"""
    if constraint.numeric_value is None:
        return False
    
    # 選択肢から数値を抽出
    choice_nums = re.findall(r'(\d+(?:\.\d+)?)', choice.text)
    if not choice_nums:
        return False
    
    for num_str in choice_nums:
        try:
            choice_val = float(num_str)
        except ValueError:
            continue
        
        cval = constraint.numeric_value
        comp = constraint.comparison
        
        if comp == '>' and constraint.negated and choice_val > cval:
            return True
        if comp == '<' and constraint.negated and choice_val < cval:
            return True
        if comp == '=' and constraint.negated and abs(choice_val - cval) < 0.01:
            return True
    
    return False


def _check_mutual_exclusion(choices: List[ChoiceCross]) -> None:
    """選択肢間の相互排他性チェック（+Z軸）"""
    # "All of the above" / "None of the above" パターン
    for ch in choices:
        text_lower = ch.text.lower().strip()
        if text_lower in ('all of the above', 'all of these', 'all are correct'):
            # 他の選択肢が相互矛盾してたらこれは偽
            pass
        elif text_lower in ('none of the above', 'none of these', 'none are correct'):
            pass
    
    # 数値選択肢: 問題文が「ちょうどN」なら他の数値は消去
    numeric_choices = []
    for ch in choices:
        nums = re.findall(r'^[\s]*(-?\d+(?:\.\d+)?)\s*$', ch.text.strip())
        if nums:
            numeric_choices.append((ch, float(nums[0])))
    
    # 重複選択肢の検出（まったく同じ主張 → 片方消去はしない）


def _negation_mismatch(constraint: Constraint, choice: ChoiceCross) -> float:
    """否定の不一致スコア"""
    overlap = _compute_keyword_overlap(constraint.keywords, choice.keywords)
    if overlap < 0.15:
        return 0.0
    
    choice_lower = choice.text.lower()
    choice_negated = any(neg in choice_lower.split() for neg in NEGATION_WORDS)
    
    # 制約が否定で選択肢が肯定（またはその逆）→ 矛盾の可能性
    if constraint.negated != choice_negated and overlap > 0.25:
        return overlap * 0.5
    
    return 0.0


def solve_mcq_by_cross_elimination(
    problem_text: str,
    choices: Dict[str, str],
    knowledge_facts: Optional[List[dict]] = None,
    ir_dict: Optional[dict] = None,
) -> Optional[Tuple[str, float, str]]:
    """
    Cross構造消去法でMCQを解く。
    
    Returns:
        (answer_label, confidence, method_description) or None
    """
    if not choices or len(choices) < 2:
        return None
    
    # Step 1: 問題文から制約を抽出 (+X軸)
    constraints = _extract_constraints(problem_text)
    if not constraints:
        return None
    
    # Step 2: 各選択肢のcross構造を構築 (-X軸)
    choice_crosses = []
    for label, text in choices.items():
        choice_crosses.append(_build_choice_cross(label, text))
    
    # Step 3: 制約→選択肢のマッチング (+Y支持, -Y矛盾)
    for constraint in constraints:
        for cc in choice_crosses:
            # キーワード重複 → 支持スコア
            overlap = _compute_keyword_overlap(constraint.keywords, cc.keywords)
            cc.support_score += overlap
            
            # 否定不一致 → 矛盾スコア
            neg_score = _negation_mismatch(constraint, cc)
            cc.contradiction_score += neg_score
            
            # 数値矛盾チェック
            if _check_numeric_contradiction(constraint, cc):
                cc.contradiction_score += 0.5
    
    # Step 4: knowledge_facts からの追加支持
    if knowledge_facts:
        fact_keywords = set()
        for f in knowledge_facts:
            if isinstance(f, dict):
                text = f.get('summary', '') or f.get('plain', '') or ''
            else:
                text = str(f)
            fact_keywords.update(_tokenize(text))
        
        for cc in choice_crosses:
            fact_overlap = _compute_keyword_overlap(fact_keywords, cc.keywords)
            cc.support_score += fact_overlap * 2.0  # facts は重み2倍
    
    # Step 5: 相互排他性チェック (+Z軸)
    _check_mutual_exclusion(choice_crosses)
    
    # Step 6: 最終スコア計算 → 消去 → 生存者選択
    for cc in choice_crosses:
        # net_score = support - contradiction
        cc.support_score -= cc.contradiction_score
    
    # ソート: support_score降順
    choice_crosses.sort(key=lambda x: x.support_score, reverse=True)
    
    best = choice_crosses[0]
    second = choice_crosses[1] if len(choice_crosses) > 1 else None
    
    # 信頼度: best と second の差
    gap = best.support_score - (second.support_score if second else 0)
    
    # 閾値チェック
    if gap < 0.05:  # 差が小さすぎる → 判断不能
        return None
    
    # confidence: gapに基づく (0.3 ~ 0.8)
    confidence = min(0.80, 0.30 + gap * 2.0)
    
    n_choices = len(choices)
    method = (
        f"cross_elimination:best={best.label}"
        f"(support={best.support_score:.3f},gap={gap:.3f},"
        f"constraints={len(constraints)},choices={n_choices})"
    )
    
    return best.label, confidence, method
