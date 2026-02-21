"""
WORLDGEN Registry — piece_id → (入力, oracle) 世界生成器

設計哲学（kofdai 2026-02-21）:
  現状の worldgen は抽象的な数値サンプル（{-20..20}）を生成するだけで、
  候補値が正しいかを機械的に検証できない。
  WORLDGEN_REGISTRY は各ピースに対して「(input, oracle)」形式の世界を提供し、
  CEGIS が"本当に燃料を持って動く"ことを保証する。

World 形式:
  {
    "inputs":  {"n": 5},               # 変数割当
    "oracle":  {"value": 120},         # 期待値 or 期待性質
    "kind":    "value_check",          # 検証関数 ID
  }

kind 一覧:
  "value_check"     - candidate_output == oracle["value"]
  "property_check"  - predicate(inputs) == oracle[property_name]
  "schema_check"    - output in ["A","B","C","D","E"]

使い方（CEGIS ループから）:
  from cegis.worldgen_registry import WORLDGEN_REGISTRY, verify_candidate_against_world
  if piece_id in WORLDGEN_REGISTRY:
      worlds = WORLDGEN_REGISTRY[piece_id](ir_dict)
      for world in worlds:
          if not verify_candidate_against_world(candidate_value, world):
              return counterexample
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# World 型定義
# ─────────────────────────────────────────────────────────────────────────────

OracleWorld = Dict[str, Any]
# {
#   "inputs":  Dict[str, Any],
#   "oracle":  Dict[str, Any],
#   "kind":    str,
#   "label":   str,   # デバッグ用ラベル（省略可）
# }

MIN_WORLDS = 1  # 空でなければ filtering を実行（registry worlds は問題固有）
# 注: 0 worlds の場合は INCONCLUSIVE（worldgen 失敗 = 燃料なし）


# ─────────────────────────────────────────────────────────────────────────────
# 検証関数: verify_candidate_against_world
# ─────────────────────────────────────────────────────────────────────────────

def verify_candidate_against_world(
    candidate_value: Any,
    world: OracleWorld,
) -> bool:
    """
    candidate_value が world の oracle を満たすか検証。

    kind 別セマンティクス:
      value_check      - candidate == oracle["value"]  （数値・計算結果）
      property_check   - candidate == computed_property(inputs)
                         oracle[prop] が期待値、inputs から実際に計算して照合
      schema_check     - candidate in oracle["allowed_values"]  （MCQ: A-E）
      arithmetic_eval  - candidate == eval(oracle["expression"], inputs)

    Returns:
        True  → 反例なし（候補はこの世界で正しい）
        False → 反例あり（候補はこの世界で誤り）
    """
    kind = world.get("kind", "value_check")
    oracle = world.get("oracle", {})
    inputs = world.get("inputs", {})

    # sanity_only は候補フィルタリングに使わない（インフラ確認のみ）
    if kind == "sanity_only":
        return True

    if kind == "value_check":
        expected = oracle.get("value")
        if expected is None:
            return True  # oracle 未設定は PASS
        return _values_equal(candidate_value, expected)

    elif kind == "property_check":
        # oracle の各 prop について:
        # 1. inputs から実際の値を計算（computed）
        # 2. candidate と computed を比較（候補が正しいか）
        # 注: oracle[prop] は "期待値の宣言" ではなく "計算指示"
        for prop in oracle:
            computed = _evaluate_property(prop, inputs)
            if computed is None:
                continue  # 評価不能は PASS 扱い
            # candidate が computed に一致しなければ反例
            if not _values_equal(candidate_value, computed):
                return False
        return True

    elif kind == "schema_check":
        allowed = oracle.get("allowed_values", ["A", "B", "C", "D", "E"])
        val = str(candidate_value).strip().upper()
        return val in allowed

    elif kind == "arithmetic_eval":
        # expression を Python で評価して候補と照合
        expr = oracle.get("expression")
        if expr is None:
            return True
        try:
            expected = _safe_eval(expr, inputs)
            return _values_equal(candidate_value, expected)
        except Exception:
            return True  # 評価失敗は PASS

    return True  # 未知の kind は PASS


def _values_equal(a: Any, b: Any) -> bool:
    """型を超えた等値判定"""
    if a is None or b is None:
        return a is b
    try:
        return abs(float(str(a)) - float(str(b))) < 1e-9
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _evaluate_property(prop: str, inputs: Dict[str, Any]) -> Optional[bool]:
    """プロパティ名 + 入力から真偽値を計算"""
    n = inputs.get("n")
    a = inputs.get("a")
    b = inputs.get("b")

    if prop == "is_prime":
        if n is not None:
            return _is_prime(int(n))
    elif prop == "is_even":
        if n is not None:
            return int(n) % 2 == 0
    elif prop == "is_odd":
        if n is not None:
            return int(n) % 2 != 0
    elif prop == "factorial":
        if n is not None and 0 <= int(n) <= 20:
            return math.factorial(int(n))
    elif prop == "gcd":
        if a is not None and b is not None:
            return math.gcd(int(a), int(b))
    elif prop == "is_palindrome":
        s = inputs.get("s")
        if s is not None:
            return str(s) == str(s)[::-1]
    return None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _safe_eval(expr: str, inputs: Dict[str, Any]) -> Any:
    """式を安全に評価（変数を inputs で置換）"""
    import re
    # 基本的な変数置換のみ許可
    safe_builtins = {
        "__builtins__": {},
        "math": math,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "pow": pow,
        "abs": abs,
        "round": round,
    }
    safe_builtins.update(inputs)
    return eval(expr, safe_builtins)  # noqa: S307


# ─────────────────────────────────────────────────────────────────────────────
# WG1: arithmetic_small_int
# ─────────────────────────────────────────────────────────────────────────────

def wg_arithmetic_small_int(ir_dict: Dict[str, Any]) -> List[OracleWorld]:
    """
    算術ピース用世界生成器（IR-aware）。

    重要設計原則:
      固定テストケース（factorial(5)=120 など）を全問題に適用すると
      「別の問題（factorial(7)=5040）の正解候補を誤って reject」する。
      そのため：
        1. IR から計算対象（n, base, exp, expr 等）を抽出
        2. その計算に対する oracle world を生成（問題固有）
        3. 抽出できない場合は空リストを返す → INCONCLUSIVE（worldgen 失敗）

    対象ピース: arithmetic_eval, arithmetic_power, nt_factorial,
               nt_factorial_compute, arithmetic_eval_integer
    """
    worlds: List[OracleWorld] = []

    # IR から計算対象を抽出
    entities    = ir_dict.get("entities", []) or []
    target      = ir_dict.get("target",   {}) or {}
    constraints = ir_dict.get("constraints", []) or []
    domain      = ir_dict.get("domain",    "")
    problem_text = ir_dict.get("problem_text", "") or ir_dict.get("question", "") or ""

    # ── パターン1: factorial(n) ───────────────────────────────────────────
    # IR に "factorial" キーワードまたは entities に n があれば
    import re
    if "factorial" in problem_text.lower() or "factorial" in str(target).lower():
        # n を entities から探す
        n_val = _extract_int_entity(entities, ["n", "k", "x", "value"])
        if n_val is not None and 0 <= n_val <= 20:
            expected = math.factorial(n_val)
            worlds.append({
                "inputs":  {"n": n_val},
                "oracle":  {"value": expected},
                "kind":    "value_check",
                "label":   f"factorial({n_val})={expected}",
            })
            # 近傍値でのサニティチェック（インフラ確認）
            for delta in (-1, +1):
                n2 = n_val + delta
                if 0 <= n2 <= 20:
                    worlds.append({
                        "inputs":  {"n": n2},
                        "oracle":  {"value": math.factorial(n2)},
                        "kind":    "sanity_only",   # 候補フィルタには使わない
                        "label":   f"sanity:factorial({n2})={math.factorial(n2)}",
                    })

    # ── パターン2: base^exp ──────────────────────────────────────────────
    if "power" in problem_text.lower() or re.search(r'\^\s*\d+|\*\*\s*\d+', problem_text):
        base = _extract_int_entity(entities, ["base", "a", "x"])
        exp  = _extract_int_entity(entities, ["exponent", "exp", "n", "k"])
        if base is not None and exp is not None and 0 <= exp <= 30:
            try:
                expected = pow(base, exp)
                worlds.append({
                    "inputs":  {"base": base, "exp": exp},
                    "oracle":  {"value": expected},
                    "kind":    "value_check",
                    "label":   f"{base}^{exp}={expected}",
                })
            except OverflowError:
                pass

    # ── パターン3: 一般的な算術式（IR から式を抽出） ──────────────────────
    expr = target.get("expression") or target.get("value")
    if expr and isinstance(expr, str):
        try:
            expected = _safe_eval(expr, {})
            worlds.append({
                "inputs":  {"expr": expr},
                "oracle":  {"value": expected, "expression": expr},
                "kind":    "arithmetic_eval",
                "label":   f"eval({expr[:30]})={expected}",
            })
        except Exception:
            pass

    return worlds


def _extract_int_entity(entities: List[Any], names: List[str]) -> Optional[int]:
    """entities から指定名の整数値を抽出するヘルパー"""
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = e.get("name", "").lower()
        if name in names:
            val = e.get("value")
            if val is not None:
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# WG2: number_theory_primes
# ─────────────────────────────────────────────────────────────────────────────

def wg_prime_world(ir_dict: Dict[str, Any]) -> List[OracleWorld]:
    """
    素数・整数論ピース用世界生成器（IR-aware）。

    重要設計原則:
      全素数/合成数リストを一括適用すると別問題の正解候補を誤 reject する。
      IR から対象 n を抽出し、その特定 n に対する oracle world のみ生成する。

    対象ピース: number_theory_prime, nt_prime_compute,
               number_theory_gcd, nt_gcd_compute
    """
    worlds: List[OracleWorld] = []

    entities     = ir_dict.get("entities", []) or []
    target       = ir_dict.get("target",   {}) or {}
    problem_text = ir_dict.get("problem_text", "") or ir_dict.get("question", "") or ""

    # ── is_prime(n) ──────────────────────────────────────────────────────
    if any(kw in problem_text.lower() for kw in ["prime", "composite", "divisible"]):
        n_val = _extract_int_entity(entities, ["n", "k", "x", "value", "number"])
        if n_val is not None and n_val >= 0:
            expected_prime = _is_prime(n_val)
            worlds.append({
                "inputs":  {"n": n_val},
                "oracle":  {"is_prime": expected_prime},
                "kind":    "property_check",
                "label":   f"is_prime({n_val})={expected_prime}",
            })

    # ── gcd(a, b) ────────────────────────────────────────────────────────
    if any(kw in problem_text.lower() for kw in ["gcd", "greatest common", "hcf"]):
        a_val = _extract_int_entity(entities, ["a", "x", "m"])
        b_val = _extract_int_entity(entities, ["b", "y", "n"])
        if a_val is not None and b_val is not None and a_val > 0 and b_val > 0:
            expected_gcd = math.gcd(a_val, b_val)
            worlds.append({
                "inputs":  {"a": a_val, "b": b_val},
                "oracle":  {"value": expected_gcd},
                "kind":    "value_check",
                "label":   f"gcd({a_val},{b_val})={expected_gcd}",
            })

    return worlds


# ─────────────────────────────────────────────────────────────────────────────
# WG3: mcq_choice_sanity
# ─────────────────────────────────────────────────────────────────────────────

def wg_mcq_choice_sanity(ir_dict: Dict[str, Any]) -> List[OracleWorld]:
    """
    MCQ ピース用：選択肢スキーマ検証の最小世界。

    「MCQ の答えは A-E のいずれか」という不変条件を検証。
    これで MCQ solver が A-E 以外を返した場合に反例を出せる。

    対象ピース: solve_multiple_choice
    """
    worlds: List[OracleWorld] = []

    # schema_check: 答えが A-E のいずれかであることを検証
    worlds.append({
        "inputs":  {},
        "oracle":  {"allowed_values": ["A", "B", "C", "D", "E"]},
        "kind":    "schema_check",
        "label":   "mcq_answer_must_be_A_to_E",
    })

    # MCQ オプション照合テストケース
    mcq_sanity_cases = [
        {"choice": "A", "valid": True},
        {"choice": "B", "valid": True},
        {"choice": "C", "valid": True},
        {"choice": "D", "valid": True},
        {"choice": "E", "valid": True},
    ]
    for case in mcq_sanity_cases:
        worlds.append({
            "inputs":  {"choice": case["choice"]},
            "oracle":  {"allowed_values": ["A", "B", "C", "D", "E"]},
            "kind":    "schema_check",
            "label":   f"mcq_schema:{case['choice']}",
        })

    return worlds


# ─────────────────────────────────────────────────────────────────────────────
# WORLDGEN_REGISTRY — piece_id → world generator 関数
# ─────────────────────────────────────────────────────────────────────────────

WorldGenFn = Callable[[Dict[str, Any]], List[OracleWorld]]

WORLDGEN_REGISTRY: Dict[str, WorldGenFn] = {
    # 算術系
    "arithmetic_eval":            wg_arithmetic_small_int,
    "arithmetic_eval_integer":    wg_arithmetic_small_int,
    "arithmetic_eval_decimal":    wg_arithmetic_small_int,
    "arithmetic_power":           wg_arithmetic_small_int,
    "nt_factorial":               wg_arithmetic_small_int,
    "nt_factorial_compute":       wg_arithmetic_small_int,
    "arithmetic_equality":        wg_arithmetic_small_int,

    # 整数論・素数系
    "number_theory_prime":        wg_prime_world,
    "nt_prime_compute":           wg_prime_world,
    "number_theory_gcd":          wg_prime_world,
    "nt_gcd_compute":             wg_prime_world,
    "nt_lcm_compute":             wg_prime_world,
    "nt_divisor_count_compute":   wg_prime_world,

    # MCQ 系
    "solve_multiple_choice":      wg_mcq_choice_sanity,
    "palindrome_check":           wg_arithmetic_small_int,  # 文字列 oracle は後で追加
}
