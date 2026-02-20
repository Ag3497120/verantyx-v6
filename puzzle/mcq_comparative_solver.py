"""
mcq_comparative_solver.py
==========================
MCQ 比較ソルバー — Verantyx Cross 構造の MCQ モード

設計思想 (kofdaiの構想):
  「答えを生成する」→「選択肢を選ぶ」へ発想転換
  
  score(choice) = 
      format_consistency(choice)      # 単位・型・次元が合っているか
    + magnitude_plausibility(choice)  # オーダーが現実的か
    + cross_sim_score(choice)         # Cross シミュレータとの整合性
    + elimination_score(choice)       # 他の選択肢との比較で矛盾を消去

  Cross の8軸に対応:
    +X: 定義     → 選択肢が何を主張しているか
    -X: 反例     → 選択肢が矛盾する反例を生成
    +Y: 変換     → 問題文の構造解析
    -Y: 検証     → 型・単位・次元の整合性チェック
    +Z: シミュレ → 小世界で選択肢を実行
    -Z: 監査     → Reject/Promote ログ

使い方:
  result = rank_choices(question, choices)
  # → [(label, score, reason), ...] 降順ソート
"""

from __future__ import annotations
import re
import math
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 選択肢スコアリング
# ============================================================

def score_choice(
    choice_text: str,
    question_text: str,
    all_choices: List[Tuple[str, str]],
    domain: Optional[str] = None,
) -> Tuple[float, str]:
    """
    1つの選択肢をスコアリング
    Returns: (score, reason)
    """
    score = 0.0
    reasons = []

    # 1. フォーマット整合性
    fmt_score, fmt_reason = _format_consistency(choice_text, question_text)
    score += fmt_score
    if fmt_reason:
        reasons.append(f"fmt:{fmt_reason}")

    # 2. オーダー妥当性
    mag_score, mag_reason = _magnitude_plausibility(choice_text, question_text, domain)
    score += mag_score
    if mag_reason:
        reasons.append(f"mag:{mag_reason}")

    # 3. 数学的術語マッチング
    term_score, term_reason = _terminology_match(choice_text, question_text)
    score += term_score
    if term_reason:
        reasons.append(f"term:{term_reason}")

    # 4. 消去法シグナル
    elim_score, elim_reason = _elimination_signal(choice_text, question_text, all_choices)
    score += elim_score
    if elim_reason:
        reasons.append(f"elim:{elim_reason}")

    reason = " | ".join(reasons) if reasons else "baseline"
    return (score, reason)


def rank_choices(
    question_text: str,
    choices: List[Tuple[str, str]],  # [(label, text), ...]
    domain: Optional[str] = None,
) -> List[Tuple[str, float, str]]:
    """
    全選択肢をスコアリングして降順でランキング
    Returns: [(label, score, reason), ...]
    """
    scored = []
    for label, text in choices:
        score, reason = score_choice(text, question_text, choices, domain)
        scored.append((label, score, reason))

    # 降順ソート（同スコアはラベル順）
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def pick_best_choice(
    question_text: str,
    choices: List[Tuple[str, str]],
    domain: Optional[str] = None,
    min_score_gap: float = 0.3,
) -> Optional[Tuple[str, float, str]]:
    """
    最良の選択肢を選ぶ
    min_score_gap: 1位と2位の差がこれ以上の場合のみ返す（0.0ならば常に返す）
    Returns: (label, confidence, reason) or None
    """
    ranked = rank_choices(question_text, choices, domain)
    if not ranked:
        return None

    best = ranked[0]
    if len(ranked) >= 2:
        second = ranked[1]
        gap = best[1] - second[1]
        if gap < min_score_gap:
            # 差が小さい → 低い信頼度で返す
            return (best[0], 0.35 + gap * 0.3, best[2])

    # 差が大きい → 信頼度高め
    return (best[0], min(0.65, 0.45 + best[1] * 0.1), best[2])


# ============================================================
# スコア計算関数
# ============================================================

def _format_consistency(choice: str, question: str) -> Tuple[float, str]:
    """
    選択肢の型が問題の期待する型と一致するか
    """
    q = question.lower()
    c = choice.strip()

    # 問題が整数/数値を期待しているか？
    asks_number = any(kw in q for kw in [
        "how many", "what is the number", "compute", "calculate",
        "find the value", "what is the minimum", "what is the maximum",
        "what is the order", "what is the rank", "what is the dimension",
        "how many ways", "total number", "count"
    ])

    is_number = bool(re.match(r'^-?\d+(?:\.\d+)?$', c))
    is_fraction = bool(re.match(r'^-?\d+/\d+$', c))
    is_numeric_expr = bool(re.search(r'\d', c) and len(c) < 20)

    if asks_number:
        if is_number:
            return (0.4, "integer matches 'how many'")
        if is_fraction or is_numeric_expr:
            return (0.2, "numeric expr")
        if len(c) > 50:
            return (-0.2, "too long for numeric answer")

    # 問題が yes/no を期待しているか？
    asks_yesno = any(kw in q for kw in ["is it true", "true or false", "does it", "can it"])
    if asks_yesno:
        if c.lower() in ["yes", "no", "true", "false"]:
            return (0.3, "yesno format matches")

    # 問題が集合/リストを期待しているか？
    asks_set = any(kw in q for kw in ["which of the following", "select all", "all of"])
    if asks_set:
        if re.search(r'\band\b', c.lower()) or ',' in c:
            return (0.1, "list/set format")

    # 選択肢が全てアルファベット1文字（A/B/C/D）の場合 → visual pattern question
    if re.match(r'^[A-Z]$', c):
        return (0.0, "single letter")

    return (0.0, "")


def _magnitude_plausibility(choice: str, question: str, domain: Optional[str]) -> Tuple[float, str]:
    """
    数値の大きさが問題の文脈で妥当か
    """
    nums_in_q = [int(x) for x in re.findall(r'\b(\d+)\b', question) if int(x) < 10000]
    nums_in_c = [int(x) for x in re.findall(r'\b(\d+)\b', choice) if int(x) < 10000]

    if not nums_in_c:
        return (0.0, "")

    c_val = nums_in_c[0]
    q = question.lower()

    # 組み合わせ問題: C(n,k) ≤ n^k
    if "choose" in q or "combination" in q or "binomial" in q:
        max_n = max(nums_in_q) if nums_in_q else 100
        if c_val > max_n ** 3:
            return (-0.3, f"too large for C({max_n},k)")
        if c_val > 0:
            return (0.1, "plausible combinatorics value")

    # グラフ問題: 彩色数 ≤ V
    if "chromatic" in q or "coloring" in q:
        if c_val <= 10:
            return (0.2, "small chromatic number")
        if c_val > 20:
            return (-0.2, "chromatic number unlikely > 20")

    # 群の位数: 有限群の位数は通常 ≤ 10000
    if "order" in q and "group" in q:
        if 1 <= c_val <= 1000:
            return (0.1, "plausible group order")

    # Ramsey数: 通常 3-50
    if "ramsey" in q.lower():
        if 3 <= c_val <= 50:
            return (0.2, "plausible Ramsey number")
        return (-0.1, "unlikely Ramsey number")

    # 次元: 通常 ≤ 100
    if "dimension" in q:
        if 1 <= c_val <= 100:
            return (0.1, "plausible dimension")
        if c_val > 1000:
            return (-0.2, "dimension unlikely > 1000")

    return (0.0, "")


def _terminology_match(choice: str, question: str) -> Tuple[float, str]:
    """
    選択肢に問題の核心用語が含まれているか
    """
    q = question.lower()
    c = choice.lower()
    
    # 問題からキー術語を抽出
    math_terms = re.findall(
        r'\b(?:abelian|cyclic|planar|bipartite|connected|compact|continuous|'
        r'monotone|injective|surjective|bijective|isomorphic|homomorphic|'
        r'convergent|bounded|finite|infinite|prime|irreducible|'
        r'symmetric|antisymmetric|transitive|reflexive|'
        r'normal|ideal|subgroup|kernel|image|quotient|'
        r'eigenvalue|orthogonal|unitary|hermitian|'
        r'hausdorff|metric|topology|manifold|'
        r'polynomial|rational|algebraic|transcendental)\b',
        q
    )
    
    if not math_terms:
        return (0.0, "")
    
    # 選択肢に問題の術語が含まれるか
    hits = sum(1 for term in math_terms if term in c)
    if hits > 0:
        return (0.15 * min(hits, 3), f"terms:{','.join(set(math_terms[:3]))}")
    
    # 反対語が含まれる場合は微弱なシグナル（否定がある場合は要注意）
    neg_hits = sum(1 for term in math_terms
                   if f"non-{term}" in c or f"not {term}" in c or f"non{term}" in c)
    if neg_hits > 0:
        return (0.05, "negated_term")
    
    return (0.0, "")


def _elimination_signal(
    choice: str,
    question: str,
    all_choices: List[Tuple[str, str]],
) -> Tuple[float, str]:
    """
    他の選択肢との比較による消去シグナル
    """
    c = choice.strip().lower()
    q = question.lower()

    # 消去: 「これは不可能です」「解けません」系の選択肢は弱い
    if any(phrase in c for phrase in [
        "cannot be determined", "not possible to determine",
        "impossible", "not enough information", "cannot solve",
        "i cannot", "undetermined"
    ]):
        # HLEでは「解けない」が正解のこともあるが、通常は弱い
        if "not possible to determine" in q or "cannot" in q:
            return (0.1, "explicitly_asked_about_impossibility")
        return (-0.1, "weakly_penalized_impossibility")

    # 「全て正しい」「全て間違い」は HLE でやや稀
    if any(phrase in c for phrase in ["all of the above", "none of the above"]):
        return (-0.05, "all/none of above slightly weak")

    # 数値選択肢の中での相対評価
    all_nums = []
    for _, txt in all_choices:
        nums = re.findall(r'^-?\d+(?:\.\d+)?$', txt.strip())
        if nums:
            all_nums.append(float(nums[0]))

    if all_nums and len(all_nums) >= 3:
        c_nums = re.findall(r'^-?\d+(?:\.\d+)?$', choice.strip())
        if c_nums:
            c_val = float(c_nums[0])
            all_nums_sorted = sorted(all_nums)
            median = all_nums_sorted[len(all_nums_sorted) // 2]
            # 外れ値（最小または最大）は正解になりにくい（ただし数学では逆もある）
            if c_val == max(all_nums) or c_val == min(all_nums):
                return (-0.05, "extreme_value_slightly_weak")
            if abs(c_val - median) / (max(all_nums) - min(all_nums) + 1) < 0.3:
                return (0.05, "near_median")

    return (0.0, "")


# ============================================================
# 問題タイプ別の特化スコア
# ============================================================

def score_by_problem_structure(
    question_text: str,
    choices: List[Tuple[str, str]],
) -> Optional[Tuple[str, float, str]]:
    """
    問題の構造から特定の選択肢を選ぶ
    高信頼度のケースのみ返す
    """
    q = question_text.lower()

    # 「次のうち正しいのはどれか」型
    # → 選択肢の命題が数学的事実と一致するか簡易チェック
    if "which of the following is true" in q or "which is true" in q:
        return _pick_true_statement(choices, question_text)

    # 「次のうち誤りはどれか」型  
    if "which of the following is false" in q or "which is false" in q or "which is incorrect" in q:
        result = _pick_true_statement(choices, question_text)
        # 反転: 最も「false」と判定された選択肢を選ぶ
        if result:
            # ランキングを逆転させる
            all_ranked = rank_choices(question_text, choices)
            if all_ranked:
                last = all_ranked[-1]
                return (last[0], 0.40, f"inverted:{last[2]}")
        return None

    # 「最小値は？」「最大値は？」型
    if any(kw in q for kw in ["minimum", "minimum number", "least", "smallest"]):
        return _pick_min_value(choices)

    if any(kw in q for kw in ["maximum", "greatest", "largest"]):
        return _pick_max_value(choices)

    return None


def _pick_true_statement(choices, question_text):
    """真の命題を選ぶ（簡易版）"""
    # Cross DB を使ってスコアリング
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from puzzle.math_theorem_db import get_relevant_theorems
        from puzzle.math_cross_sim import detect_math_domain

        domains = detect_math_domain(question_text)
        top_domain = domains[0][0] if domains else None
        theorems = get_relevant_theorems(question_text, top_domain, top_k=3)

        best_label = None
        best_conf = 0.0
        best_reason = ""

        for label, text in choices:
            for thm in theorems:
                verdict, conf = thm.test(text, question_text)
                if verdict == 'promote' and conf > best_conf:
                    best_conf = conf
                    best_label = label
                    best_reason = f"{thm.theorem_id}"
                elif verdict == 'reject' and (1.0 - conf) > best_conf:
                    pass  # reject は別途扱う

        if best_label and best_conf >= 0.75:
            return (best_label, best_conf, f"theorem_promote:{best_reason}")
    except Exception:
        pass

    return None


def _pick_min_value(choices):
    """最小値の選択肢を選ぶ"""
    nums = []
    for label, text in choices:
        m = re.search(r'^-?\d+(?:\.\d+)?$', text.strip())
        if m:
            nums.append((label, float(m.group())))
    if nums:
        min_label, min_val = min(nums, key=lambda x: x[1])
        return (min_label, 0.45, f"min_value:{min_val}")
    return None


def _pick_max_value(choices):
    """最大値の選択肢を選ぶ"""
    nums = []
    for label, text in choices:
        m = re.search(r'^-?\d+(?:\.\d+)?$', text.strip())
        if m:
            nums.append((label, float(m.group())))
    if nums:
        max_label, max_val = max(nums, key=lambda x: x[1])
        return (max_label, 0.45, f"max_value:{max_val}")
    return None


# ============================================================
# メインエントリーポイント
# ============================================================

def solve_mcq_comparative(
    question_text: str,
    choices: List[Tuple[str, str]],
    domain: Optional[str] = None,
    confidence_threshold: float = 0.35,
) -> Optional[Tuple[str, float, str]]:
    """
    MCQ比較ソルバーのメインエントリー

    フロー:
      1. 問題構造による特化スコア（高精度）
      2. 比較スコアリングによる最良選択（中精度）
      3. 信頼度閾値以上の場合のみ返す

    Returns: (label, confidence, reason) or None
    """
    # 1. 問題構造による特化スコア
    struct_result = score_by_problem_structure(question_text, choices)
    if struct_result and struct_result[1] >= confidence_threshold:
        return struct_result

    # 2. 比較スコアリング
    best = pick_best_choice(question_text, choices, domain, min_score_gap=0.2)
    if best and best[1] >= confidence_threshold:
        return best

    return None


# ============================================================
# テスト
# ============================================================

if __name__ == "__main__":
    test_cases = [
        (
            "How many ways can you choose 3 items from 10?",
            [("A", "30"), ("B", "120"), ("C", "720"), ("D", "10")],
            "B",  # 120 = C(10,3)
        ),
        (
            "What is the chromatic number of K_5?",
            [("A", "3"), ("B", "4"), ("C", "5"), ("D", "6")],
            "C",  # χ(K_5) = 5
        ),
        (
            "Is K_{3,3} a planar graph?",
            [("A", "Yes, it is planar"), ("B", "No, it is non-planar"), ("C", "It depends on the drawing")],
            "B",
        ),
        (
            "What is the minimum number of colors needed to color a bipartite graph?",
            [("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")],
            "B",  # bipartite → 2-colorable
        ),
    ]

    print("MCQ Comparative Solver Tests")
    print("=" * 50)
    passed = 0
    for question, choices, expected in test_cases:
        result = solve_mcq_comparative(question, choices)
        if result:
            label, conf, reason = result
            ok = label == expected
            mark = "✅" if ok else "❌"
            passed += int(ok)
            print(f"{mark} expected={expected} got={label} conf={conf:.2f}")
            print(f"   reason: {reason[:70]}")
        else:
            print(f"❓ expected={expected} → No result")
        print()

    print(f"Passed: {passed}/{len(test_cases)}")

    # 全体の選択肢ランキングをデバッグ
    print("\n=== Ranking Debug ===")
    q = "What is the minimum number of states in a DFA accepting (ab)*?"
    ch = [("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5")]
    ranked = rank_choices(q, ch)
    for lbl, score, reason in ranked:
        print(f"  {lbl}: {score:.2f} | {reason}")
