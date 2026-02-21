"""
mcq_reasoning_executor.py
==========================
600B 推論型 → Verantyx Executor へのルーティング

フロー:
  1. ReasoningTypeClassifier が推論型を特定
  2. 推論型に応じた Executor 戦略を選択
  3. 各選択肢を評価して正答を返す

設計原則:
  - INCONCLUSIVE is safe（証明できなければ回答しない）
  - 計算で検証できる選択肢のみ返す
  - 知識系/形式証明系はINCONCLUSIVE
"""

import re
from typing import Dict, Optional, Tuple, List


# ── 推論型別 Executor 戦略 ──────────────────────────────────────────────────

def _execute_verify_property(
    stem: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    各選択肢に性質チェックを適用して True のものを返す。

    対応プロパティ:
      - is_prime
      - is_divisible_by
      - is_palindrome
      - is_perfect_square
      - is_even / is_odd
    """
    import sympy

    stem_lower = stem.lower()

    # どの性質チェックをするか特定
    if "prime" in stem_lower:
        def check(val_str):
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                return sympy.isprime(n)
            except Exception:
                return None

    elif "divisible" in stem_lower:
        # "divisible by X and Y" → X, Y を抽出
        nums = [int(x) for x in re.findall(r'\b(\d+)\b', stem_lower) if int(x) > 1]
        if not nums:
            return None
        def check(val_str):
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                return all(n % d == 0 for d in nums)
            except Exception:
                return None

    elif "palindrome" in stem_lower:
        def check(val_str):
            s = val_str.strip().lower()
            return s == s[::-1]

    elif "perfect square" in stem_lower or "square number" in stem_lower:
        def check(val_str):
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                if n < 0:
                    return False
                return int(n ** 0.5) ** 2 == n
            except Exception:
                return None

    elif re.search(r'\beven\b', stem_lower) and "not" not in stem_lower:
        def check(val_str):
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                return n % 2 == 0
            except Exception:
                return None

    elif re.search(r'\bodd\b', stem_lower):
        def check(val_str):
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                return n % 2 == 1
            except Exception:
                return None

    else:
        return None  # 対応する性質チェックなし

    # 各選択肢に適用
    verified = []
    for label, option_text in sorted(choices.items()):
        result = check(option_text)
        if result is True:
            verified.append(label)

    if len(verified) == 1:
        return verified[0], 0.95, f"verify_property:{stem_lower[:30]}"
    return None


def _execute_compute_and_match(
    stem: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    stem から値を計算し、一致する選択肢を返す。

    対応計算:
      - factorial (n!)
      - power (x^n, x**n)
      - simple arithmetic
      - GCD / LCM
    """
    stem_lower = stem.lower()

    computed_val = None
    method_name = ""

    # 階乗
    m = re.search(r'(\d+)\s*!|factorial\s+of\s+(\d+)|(\d+)\s+factorial', stem_lower)
    if m:
        n = int(next(g for g in m.groups() if g is not None))
        if n <= 20:
            import math
            computed_val = math.factorial(n)
            method_name = f"factorial({n})"

    # べき乗
    if computed_val is None:
        m = re.search(r'(\d+)\s*\*{2}\s*(\d+)|(\d+)\s*\^\s*(\d+)', stem_lower)
        if m:
            gs = m.groups()
            if gs[0]:
                base, exp_ = int(gs[0]), int(gs[1])
            else:
                base, exp_ = int(gs[2]), int(gs[3])
            if exp_ <= 30:
                computed_val = base ** exp_
                method_name = f"{base}^{exp_}"

    # GCD
    if computed_val is None and 'gcd' in stem_lower:
        nums = [int(x) for x in re.findall(r'\b(\d+)\b', stem_lower) if int(x) > 1]
        if len(nums) >= 2:
            import math
            computed_val = math.gcd(nums[0], nums[1])
            method_name = f"gcd({nums[0]},{nums[1]})"

    # 簡単な算術式
    if computed_val is None:
        m = re.search(r'what\s+is\s+([\d\s\+\-\*/\(\)]+)\??$', stem_lower)
        if m:
            expr = m.group(1).strip()
            if all(c in '0123456789+-*/ ()' for c in expr) and len(expr) < 30:
                try:
                    computed_val = eval(expr)  # noqa: S307
                    method_name = f"eval({expr})"
                except Exception:
                    pass

    if computed_val is None:
        return None

    # 一致する選択肢を探す
    for label, option_text in sorted(choices.items()):
        opt_nums = re.findall(r'-?\d+(?:\.\d+)?', option_text)
        for num_str in opt_nums:
            try:
                opt_val = float(num_str)
                if abs(opt_val - float(computed_val)) < 1e-6:
                    return label, 0.95, f"compute_and_match:{method_name}={computed_val}"
            except ValueError:
                pass

    return None


def _execute_elimination(
    stem: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    「〜でないもの」「〜できないもの」→ 各選択肢でプロパティを評価し
    満たさない唯一の選択肢を返す（否定型MCQ対応）。
    """
    stem_lower = stem.lower()

    negated = bool(re.search(
        r'\bnot\b|\bcannot\b|\bimpossible\b|\bincorrect\b|\bfalse\b',
        stem_lower
    ))
    if not negated:
        return None

    # 否定キーワードを除いた stem で property チェックを実施
    stem_pos = re.sub(
        r'\bnot\b|\bcannot\b|\bimpossible\b|\bincorrect\b|\bfalse\b',
        '', stem_lower
    ).strip()

    # 各選択肢ごとにプロパティを評価（True/False/None）
    import sympy
    stem_p = stem_pos

    def prop_check(val_str: str):
        if "prime" in stem_p:
            try:
                n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                return sympy.isprime(n)
            except Exception:
                return None
        if "divisible" in stem_p:
            nums = [int(x) for x in re.findall(r'\b(\d+)\b', stem_p) if int(x) > 1]
            if nums:
                try:
                    n = int(re.sub(r'[^0-9\-]', '', val_str.split()[0]))
                    return all(n % d == 0 for d in nums)
                except Exception:
                    return None
        if "palindrome" in stem_p:
            s = val_str.strip().lower()
            return s == s[::-1]
        return None

    results = {
        label: prop_check(option_text)
        for label, option_text in sorted(choices.items())
    }

    # None は評価不能
    clear = {k: v for k, v in results.items() if v is not None}
    if not clear:
        return None

    # NOT → False の選択肢が唯一なら正答
    false_labels = [k for k, v in clear.items() if not v]
    true_labels  = [k for k, v in clear.items() if v]

    if len(false_labels) == 1:
        return false_labels[0], 0.90, f"elimination:not_property:{stem_p[:25]}"

    # Fallback: True が複数あり False が1つ（部分評価）
    if len(false_labels) == 1 and len(true_labels) >= 2:
        return false_labels[0], 0.85, "elimination:not_property:partial"

    return None


# ── メイン関数 ─────────────────────────────────────────────────────────────

def execute_mcq_by_reasoning(
    stem: str,
    choices: Dict[str, str],
    use_600b_classifier: bool = True,
    confidence_threshold: float = 0.08,
) -> Optional[Tuple[str, float, str]]:
    """
    600B 推論型分類 → Verantyx Executor で MCQ を解く。

    Args:
        stem: 問題文（選択肢なし）
        choices: {"A": "...", "B": "...", ...}
        use_600b_classifier: True なら推論型分類器を使用
        confidence_threshold: この値未満の信頼度ならINCONCLUSIVE

    Returns:
        (answer_label, confidence, method) or None (INCONCLUSIVE)
    """
    if use_600b_classifier:
        try:
            from knowledge.reasoning_type_classifier import get_classifier
            clf = get_classifier()
            computable, rtype, conf = clf.is_computable(stem)

            if computable:
                # 推論型に応じてルーティング（優先）
                if rtype == "verify_property":
                    result = _execute_verify_property(stem, choices)
                    if result:
                        return result
                elif rtype == "compute_and_match":
                    result = _execute_compute_and_match(stem, choices)
                    if result:
                        return result
                elif rtype == "elimination":
                    result = _execute_elimination(stem, choices)
                    if result:
                        return result

            # 信頼度が低い or 計算不能と判定されてもフォールバックで全戦略を試す
            # （ただし formal_proof / factual_lookup 系は試さない）
            if rtype not in ("formal_proof", "factual_lookup"):
                for strategy in [_execute_verify_property, _execute_compute_and_match, _execute_elimination]:
                    result = strategy(stem, choices)
                    if result:
                        return result

            # formal_proof / factual_lookup → INCONCLUSIVE（計算不能）
            return None

        except Exception:
            pass  # 600B 利用不可の場合は下のフォールバックへ

    # 600B なしフォールバック: 全戦略を試す
    for strategy in [_execute_verify_property, _execute_compute_and_match, _execute_elimination]:
        result = strategy(stem, choices)
        if result:
            return result

    return None  # INCONCLUSIVE


# ── テスト ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    tests = [
        {
            "stem": "Which of the following is a prime number?",
            "choices": {"A": "4", "B": "6", "C": "7", "D": "9"},
            "correct": "C",
        },
        {
            "stem": "What is 5! (5 factorial)?",
            "choices": {"A": "60", "B": "100", "C": "120", "D": "24"},
            "correct": "C",
        },
        {
            "stem": "Which number is divisible by both 3 and 4?",
            "choices": {"A": "10", "B": "14", "C": "24", "D": "26"},
            "correct": "C",
        },
        {
            "stem": "Which of the following is NOT a prime number?",
            "choices": {"A": "7", "B": "11", "C": "9", "D": "13"},
            "correct": "C",
        },
        {
            "stem": "What is 2**10?",
            "choices": {"A": "512", "B": "1024", "C": "256", "D": "2048"},
            "correct": "B",
        },
        {
            "stem": "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?",
            "choices": {"A": "Egalitarian Dominance", "B": "Non-Sadism", "C": "Quality Addition", "D": "Weak Non-Elitism"},
            "correct": "D",
        },
    ]

    print("=== MCQ Reasoning Executor テスト ===\n")
    correct_count = 0
    for t in tests:
        result = execute_mcq_by_reasoning(t["stem"], t["choices"])
        if result:
            label, conf, method = result
            ok = "✅" if label == t["correct"] else "❌"
            correct_count += (label == t["correct"])
        else:
            ok = "✅" if t["correct"] == "INCONCLUSIVE" else "⚪"
            label, conf, method = "INCONCLUSIVE", 0.0, "none"
        print(f"{ok} Q: {t['stem'][:55]}")
        print(f"   → {label} (conf={conf:.2f}) [{method}]  expected={t['correct']}")
        print()

    print(f"Score: {correct_count}/{len(tests)}")
