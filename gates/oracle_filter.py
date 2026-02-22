"""
oracle_filter.py  ← Step A: oracle filtering 強化
"oracle empty → INCONCLUSIVE" を徹底し、wrong proved を根絶する。

差し込み先: CrossSim / CEGIS の手前（verify_spec を受け取る全箇所）

使い方:
    from gates.oracle_filter import check_oracle_before_verify, OracleFilterResult

    result = check_oracle_before_verify(verify_spec, ir_entities)
    if not result.ok:
        return AuditBundle(status="inconclusive", reason=result.reason)
    # → 通過後にCEGIS/CrossSimを呼ぶ
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class OracleFilterResult:
    ok: bool
    reason: str = ""   # ok=False のときの reason（audit_logに渡す）


# ---------------------------------------------------------------------------
# 1. vacuous oracle の定義（GateCの再利用 + 追加パターン）
# ---------------------------------------------------------------------------

VACUOUS_ORACLES = {
    "", "True", "true", "None", "none",
    "isinstance(x, int)", "isinstance(x, str)",
    "type(x) == int", "type(x) == str",
    "pass", "...", "lambda x: True", "any", "all",
}

# oracle に「具体的な演算 or ライブラリ参照」が必要
ORACLE_HAS_OPERATION = re.compile(
    r"[a-zA-Z_]\w*\s*\("     # 関数呼び出し
    r"|[+\-*/%]"              # 算術演算子
    r"|==|!=|<=|>=|<|>"       # 比較演算子
    r"|math\.|sympy\.|numpy\.|scipy\."
    r"|\*\*"                  # 冪乗
)

# MCQ schema-only oracle のパターン（"A" | "B" | ... を返すだけ）
MCQ_SCHEMA_ONLY = re.compile(r"^['\"]?[A-E]['\"]?$")


# ---------------------------------------------------------------------------
# 2. entity バインディングチェック
#    oracle に言及されている変数が entities に存在するか
# ---------------------------------------------------------------------------

def _extract_identifiers(oracle: str) -> set[str]:
    """oracle文字列から識別子を抽出する（単純な正規表現）。"""
    return set(re.findall(r"\b([a-zA-Z_]\w*)\b", oracle))


PYTHON_BUILTINS = {
    "math", "sympy", "numpy", "scipy",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "len", "range", "sum", "min", "max", "abs", "round",
    "True", "False", "None",
    "factorial", "gcd", "lcm", "sqrt", "log", "exp", "sin", "cos",
    "pow", "mod",
}


def _check_entity_binding(oracle: str, entities: list[dict]) -> str | None:
    """
    oracle内の変数名が entities["name"] に存在しない場合 → unbound_variable を返す。
    entitiesが空なら警告のみ（チェックしない）。
    """
    if not entities:
        return None  # entity 抽出なし → チェック不可、通過

    entity_names = {str(e.get("name", "")).strip() for e in entities}
    oracle_ids = _extract_identifiers(oracle)
    unbound = oracle_ids - entity_names - PYTHON_BUILTINS

    if unbound:
        return f"unbound_vars={unbound}"
    return None


# ---------------------------------------------------------------------------
# 3. 主エントリポイント
# ---------------------------------------------------------------------------

def check_oracle_before_verify(
    verify_spec: dict | None,
    ir_entities: list[dict] | None = None,
) -> OracleFilterResult:
    """
    verify_spec を CEGIS / CrossSim に渡す前に検査する。

    Args:
        verify_spec: {"oracle": "...", "constraints": [...]}
        ir_entities: IR から抽出したエンティティ（バインディングチェック用）

    Returns:
        OracleFilterResult(ok=True) → 検証に進んでいい
        OracleFilterResult(ok=False, reason=...) → INCONCLUSIVE に落とす
    """
    # verify_spec が None / 空
    if not verify_spec or not isinstance(verify_spec, dict):
        return OracleFilterResult(ok=False, reason="oracle_missing:verify_spec_none")

    oracle = str(verify_spec.get("oracle", "")).strip()
    constraints = verify_spec.get("constraints", [])

    # 1) oracle が空 / 自明
    if oracle in VACUOUS_ORACLES:
        return OracleFilterResult(ok=False, reason=f"oracle_vacuous:'{oracle}'")

    # 2) oracle が MCQ schema-only（"A" とか "B" を返すだけ）
    if MCQ_SCHEMA_ONLY.match(oracle):
        return OracleFilterResult(ok=False, reason=f"oracle_mcq_schema_only:'{oracle}'")

    # 3) oracle に演算が全くない
    if not ORACLE_HAS_OPERATION.search(oracle):
        return OracleFilterResult(ok=False, reason=f"oracle_no_operation:'{oracle}'")

    # 4) constraints が空
    if not isinstance(constraints, list) or len(constraints) == 0:
        return OracleFilterResult(ok=False, reason="oracle_no_constraints")

    # 5) constraints が型チェックのみ
    type_check_pat = re.compile(r"^isinstance\(|^type\(")
    non_trivial = [c for c in constraints if not type_check_pat.match(str(c).strip())]
    if len(non_trivial) == 0:
        return OracleFilterResult(ok=False, reason="oracle_constraints_all_type_checks")

    # 6) entity バインディングチェック（警告レベル: ok=True のまま通すが reason に記録）
    entities = ir_entities or []
    unbound = _check_entity_binding(oracle, entities)
    if unbound:
        # unbound はエラーではなく警告（entities抽出が不完全なケースもある）
        # → 通過させるが reason に記録。audit_log に渡す。
        return OracleFilterResult(ok=True, reason=f"oracle_warn_unbound:{unbound}")

    return OracleFilterResult(ok=True, reason="")


# ---------------------------------------------------------------------------
# 4. ログ用ヘルパー（pipeline_enhanced.py で audit_log に渡す）
# ---------------------------------------------------------------------------

def oracle_filter_summary(result: OracleFilterResult) -> dict:
    return {
        "oracle_filter_ok": result.ok,
        "oracle_filter_reason": result.reason,
    }
