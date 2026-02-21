"""
expert_piece_map.py — A + B: Expert→Piece直結マップ + Expert→WorldgenProfile

設計思想（kofdai 2026-02-21）:
  現状の routing は "expert → domain → piece" の間接経路で情報が落ちる。
  このモジュールは "expert → piece" を直接つなぎ、
  step6_not_reached を減らし oracle_empty を落とす。

  A: expert_id → [piece_candidates]  （直接 boost）
  B: expert_id → worldgen_profile    （oracle worldgen 選択）

自動生成ロジック:
  1. pieces の piece_id + description から "指標キーワード" を定義
  2. expert_math_tokens.json (expert → token list) と照合
  3. expert のトークンに指標キーワードが含まれる → expert → piece の辺を張る

使い方:
  from knowledge.expert_piece_map import ExpertPieceMap
  epm = ExpertPieceMap()
  piece_boosts = epm.get_piece_boosts(top_experts)  # [(piece_id, score)]
  wg_profile   = epm.get_worldgen_profile(top_experts)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ピース指標キーワード定義（手動 + 自動の組み合わせ）
# ─────────────────────────────────────────────────────────────────────────────

# piece_id → 指標トークンリスト（expert_math_tokens と照合する）
# 大文字小文字は正規化して比較
PIECE_INDICATOR_TOKENS: Dict[str, List[str]] = {
    # 算術
    "arithmetic_eval":            ["eval", "evaluate", "arithmetic", "compute", "expression", "calc"],
    "arithmetic_eval_integer":    ["integer", "eval", "arithmetic", "exact"],
    "arithmetic_power":           ["power", "exponent", "^", "pow", "base", "squared", "cubed"],
    "arithmetic_equality":        ["equality", "equation", "equal", "="],

    # 整数論
    "number_theory_prime":        ["prime", "primality", "sieve", "composite"],
    "nt_prime_compute":           ["prime", "primality", "composite", "isprime"],
    "number_theory_gcd":          ["gcd", "divisor", "greatest", "common", "euclid"],
    "nt_gcd_compute":             ["gcd", "divisor", "greatest", "common", "euclid"],
    "nt_lcm":                     ["lcm", "least", "multiple", "common"],
    "nt_lcm_compute":             ["lcm", "least", "multiple", "common"],
    "nt_factorial":               ["factorial", "n!", "!", "permutation"],
    "nt_factorial_compute":       ["factorial", "n!", "!", "permutation"],
    "nt_divisor_count_compute":   ["divisor", "factor", "count", "tau"],
    "prime_factorization":        ["factorization", "prime", "factor", "product"],

    # 組み合わせ論
    "combinatorics_permutation":  ["permutation", "arrangement", "P(n", "nPr"],
    "combinatorics_combination":  ["combination", "choose", "C(n", "nCr", "binomial"],
    "comb_binomial":              ["binomial", "coefficient", "pascal", "C(n", "choose"],
    "comb_perm_compute":          ["permutation", "nPr", "arrangement"],
    "comb_comb_compute":          ["combination", "nCr", "choose"],
    "derangement_num":            ["derangement", "!n", "subfactorial"],
    "partition_count_calc":       ["partition", "integer partition"],

    # 代数
    "algebra_solve_equation":     ["equation", "solve", "algebra", "variable", "root"],
    "algebra_solve_linear":       ["linear", "equation", "solve", "ax"],
    "algebra_evaluate_poly":      ["polynomial", "evaluate", "poly", "coefficient"],

    # 確率
    "probability_basic":          ["probability", "P(", "favorable", "total", "event"],
    "prob_basic_compute":         ["probability", "compute", "P("],
    "prob_expected_value":        ["expected", "value", "E[", "expectation", "mean"],
    "probability_expected_value": ["expected", "value", "E[", "expectation"],
    "probability_coin_flip":      ["coin", "flip", "heads", "tails"],
    "probability_dice_roll":      ["dice", "roll", "d6", "faces"],
    "probability_card_draw":      ["card", "deck", "draw", "suit"],
    "probability_multiple_events":["multiple", "events", "conditional", "independent"],

    # 幾何
    "geometry_triangle_area":     ["triangle", "area", "base", "height", "\\frac{1}{2}"],
    "geom_circle_circumference":  ["circle", "circumference", "2\\pi", "diameter"],
    "geometry_circle_area":       ["circle", "area", "\\pi r", "radius"],
    "geometry_rectangle_perimeter": ["rectangle", "perimeter", "length", "width"],
    "geometry_pythagorean":       ["pythagorean", "hypotenuse", "a^2", "right triangle"],

    # 論理
    "prop_truth_table":           ["truth", "table", "propositional", "tautology", "logic"],
    "prop_decide":                ["tautology", "satisfiable", "logic", "proposition"],
    "modal_kripke_search":        ["modal", "kripke", "possible", "worlds", "box", "diamond"],

    # 文字列
    "string_length":              ["string", "length", "characters", "strlen"],
    "palindrome_check":           ["palindrome", "reverse", "symmetric"],
    "word_count_calc":            ["word", "count", "words"],

    # MCQ
    "solve_multiple_choice":      ["choice", "option", "A)", "B)", "C)", "D)", "E)", "which"],
    "option_selector":            ["choice", "option", "select", "which", "MCQ"],

    # 行列
    "modular_mod_power":          ["modular", "mod", "congruence", "pow", "\\equiv"],
    "integer_range_enumerate":    ["range", "enumerate", "integers", "from", "to"],

    # チェス
    "chess_stockfish":            ["chess", "move", "mate", "check", "board", "position"],
}

# ─────────────────────────────────────────────────────────────────────────────
# B: worldgen_profile 定義
# ─────────────────────────────────────────────────────────────────────────────

# piece_id → worldgen_profile（B: expert→worldgen ヒント）
PIECE_WORLDGEN_PROFILE: Dict[str, Dict] = {
    "arithmetic_eval":           {"wg": "wg_arithmetic_small_int", "kind": "value_check",    "min_worlds": 1},
    "arithmetic_eval_integer":   {"wg": "wg_arithmetic_small_int", "kind": "value_check",    "min_worlds": 1},
    "arithmetic_power":          {"wg": "wg_arithmetic_small_int", "kind": "value_check",    "min_worlds": 1},
    "nt_factorial":              {"wg": "wg_arithmetic_small_int", "kind": "value_check",    "min_worlds": 1},
    "nt_factorial_compute":      {"wg": "wg_arithmetic_small_int", "kind": "value_check",    "min_worlds": 1},
    "number_theory_prime":       {"wg": "wg_prime_world",          "kind": "property_check", "min_worlds": 1},
    "nt_prime_compute":          {"wg": "wg_prime_world",          "kind": "property_check", "min_worlds": 1},
    "number_theory_gcd":         {"wg": "wg_prime_world",          "kind": "value_check",    "min_worlds": 1},
    "nt_gcd_compute":            {"wg": "wg_prime_world",          "kind": "value_check",    "min_worlds": 1},
    "solve_multiple_choice":     {"wg": "wg_mcq_choice_sanity",    "kind": "schema_check",   "min_worlds": 1},
    "option_selector":           {"wg": "wg_mcq_choice_sanity",    "kind": "schema_check",   "min_worlds": 1},
    "palindrome_check":          {"wg": "wg_arithmetic_small_int", "kind": "property_check", "min_worlds": 1},
}

# ─────────────────────────────────────────────────────────────────────────────
# D: 概念キーワード → entity 抽出ヒント
# ─────────────────────────────────────────────────────────────────────────────

# piece_id → IR entity 名のヒント（D: entity 抽出補助）
PIECE_ENTITY_HINTS: Dict[str, List[str]] = {
    "nt_factorial":           ["n", "k"],
    "nt_factorial_compute":   ["n", "k"],
    "arithmetic_power":       ["base", "exponent", "exp"],
    "number_theory_prime":    ["n", "number"],
    "nt_prime_compute":       ["n", "number"],
    "number_theory_gcd":      ["a", "b"],
    "nt_gcd_compute":         ["a", "b"],
    "nt_lcm_compute":         ["a", "b"],
    "combinatorics_combination": ["n", "r", "k"],
    "comb_comb_compute":      ["n", "r"],
    "combinatorics_permutation": ["n", "r"],
    "comb_perm_compute":      ["n", "r"],
    "prob_expected_value":    ["n", "p", "x"],
    "geometry_triangle_area": ["base", "height"],
    "geometry_pythagorean":   ["a", "b", "c"],
    "modular_mod_power":      ["a", "n", "mod", "m"],
}


# ─────────────────────────────────────────────────────────────────────────────
# ExpertPieceMap クラス
# ─────────────────────────────────────────────────────────────────────────────

class ExpertPieceMap:
    """
    Expert → Piece / WorldgenProfile の直結マップ。

    起動時に expert_math_tokens.json × PIECE_INDICATOR_TOKENS で
    自動生成する。マップは LRU キャッシュで再利用。
    """

    def __init__(self):
        self._expert_piece: Dict[str, List[Tuple[str, float]]] = {}  # expert_key → [(piece_id, score)]
        self._piece_expert: Dict[str, List[str]] = {}                # piece_id → [expert_keys]
        self._built = False

    def _build(self):
        if self._built:
            return

        # expert_math_tokens.json を読む
        tokens_path = Path(__file__).parent / "expert_math_tokens.json"
        if not tokens_path.exists():
            self._built = True
            return

        with open(tokens_path) as f:
            expert_tokens: Dict[str, List[str]] = json.load(f)

        # piece_id → indicator tokens（小文字化）
        piece_indicators: Dict[str, List[str]] = {
            pid: [t.lower() for t in tokens]
            for pid, tokens in PIECE_INDICATOR_TOKENS.items()
        }

        # expert → [piece_id] の辺を張る
        for expert_key, token_list in expert_tokens.items():
            # expert トークンを小文字セットに変換
            expert_token_set = {t.lower() for t in token_list}

            matched: List[Tuple[str, float]] = []
            for pid, indicators in piece_indicators.items():
                # overlap スコア = マッチしたインジケータ数 / インジケータ総数
                hits = sum(1 for ind in indicators if any(ind in tok for tok in expert_token_set))
                if hits > 0:
                    score = hits / len(indicators)
                    matched.append((pid, score))

            if matched:
                # スコア降順でソート
                matched.sort(key=lambda x: -x[1])
                self._expert_piece[expert_key] = matched[:10]  # top-10 pieces per expert

                # 逆引きマップ
                for pid, _ in matched[:5]:
                    self._piece_expert.setdefault(pid, []).append(expert_key)

        self._built = True

    def get_piece_boosts(
        self,
        top_experts: List[Tuple[str, float]],
        top_n: int = 10,
        min_score: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """
        A: top_experts から piece_id → boost_score を計算する。

        Args:
            top_experts: [(expert_key, sim_score), ...] from ConceptSearcher.search_top_experts()
            top_n: 返す piece 数上限
            min_score: 最低 boost スコア

        Returns:
            [(piece_id, boost_score), ...] 降順
        """
        self._build()
        piece_scores: Dict[str, float] = {}

        for expert_key, expert_score in top_experts:
            for piece_id, piece_match_score in self._expert_piece.get(expert_key, []):
                # boost = expert の類似度 × piece のマッチスコア
                combined = expert_score * piece_match_score
                piece_scores[piece_id] = piece_scores.get(piece_id, 0.0) + combined

        # ソートして top-N
        sorted_pieces = sorted(piece_scores.items(), key=lambda x: -x[1])
        return [(pid, sc) for pid, sc in sorted_pieces[:top_n] if sc >= min_score]

    def get_worldgen_profile(
        self,
        top_experts: List[Tuple[str, float]],
    ) -> Optional[Dict]:
        """
        B: top_experts から最適な worldgen_profile を返す。

        Piece boost の上位 piece の worldgen_profile を返す。
        oracle worldgen 選択を expert 活性化で決める。

        Returns:
            {"wg": "wg_prime_world", "kind": "property_check", "min_worlds": 1}
            or None (マッチなし)
        """
        self._build()
        piece_boosts = self.get_piece_boosts(top_experts, top_n=5)

        for piece_id, _ in piece_boosts:
            if piece_id in PIECE_WORLDGEN_PROFILE:
                return PIECE_WORLDGEN_PROFILE[piece_id]
        return None

    def get_entity_hints(
        self,
        top_experts: List[Tuple[str, float]],
    ) -> List[str]:
        """
        D: top_experts から entity 名のヒントを返す。

        IR entity 抽出器が迷ったときの tie-breaker として使う。
        「この問題は n と k を持つはず」というヒントを与える。

        Returns:
            ["n", "k", "base", ...]  (重複なし、優先度順)
        """
        self._build()
        piece_boosts = self.get_piece_boosts(top_experts, top_n=5)
        seen = set()
        hints: List[str] = []
        for piece_id, _ in piece_boosts:
            for hint in PIECE_ENTITY_HINTS.get(piece_id, []):
                if hint not in seen:
                    hints.append(hint)
                    seen.add(hint)
        return hints[:8]  # 最大8ヒント

    def stats(self) -> Dict:
        """デバッグ用統計"""
        self._build()
        return {
            "total_experts": len(self._expert_piece),
            "total_pieces_covered": len(self._piece_expert),
            "avg_pieces_per_expert": (
                sum(len(v) for v in self._expert_piece.values()) /
                max(len(self._expert_piece), 1)
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 軽量テキストキーワード → piece 検索（concept_dirs不要・DISABLE_CONCEPT_BOOST=1でも動作）
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

def find_pieces_by_text_keywords(
    text: str,
    top_n: int = 5,
    min_hits: int = 1,
) -> List[Tuple[str, float]]:
    """
    問題文テキストから直接 piece_id を検索する軽量バージョン。
    concept_dirs / expert_math_tokens.json を使わず、PIECE_INDICATOR_TOKENS のみを使用。
    DISABLE_CONCEPT_BOOST=1 の場合でも動作する。

    Args:
        text: 問題文
        top_n: 返す piece 数上限
        min_hits: 最低マッチ数

    Returns:
        [(piece_id, score), ...] 降順
    """
    # テキストをトークン化（小文字・英単語）
    text_lower = text.lower()
    text_tokens = set(_re.findall(r'[a-z]+', text_lower))

    piece_scores: Dict[str, float] = {}
    for pid, indicators in PIECE_INDICATOR_TOKENS.items():
        # 各インジケータについてテキストに含まれるかチェック
        hits = 0
        for ind in indicators:
            ind_lower = ind.lower()
            # 部分文字列一致 or トークン完全一致
            if ind_lower in text_lower or ind_lower in text_tokens:
                hits += 1
        if hits >= min_hits:
            # スコア = マッチ数 / インジケータ総数
            score = hits / max(len(indicators), 1)
            piece_scores[pid] = score

    sorted_pieces = sorted(piece_scores.items(), key=lambda x: -x[1])
    return sorted_pieces[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# シングルトン
# ─────────────────────────────────────────────────────────────────────────────

_epm_instance: Optional[ExpertPieceMap] = None

def get_expert_piece_map() -> ExpertPieceMap:
    global _epm_instance
    if _epm_instance is None:
        _epm_instance = ExpertPieceMap()
    return _epm_instance
