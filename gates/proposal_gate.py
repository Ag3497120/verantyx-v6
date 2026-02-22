"""
proposal_gate.py
GateA / GateB / GateC — LLM proposal の3段フィルター

GateA: 答え禁止フィールドの検出（answer/final/solution/correct/result）
GateB: 必須フィールドの存在チェック（JSONスキーマ）
GateC: verify_spec の実質性チェック（vacuous proved 根絶）

戻り値: None = 通過, str = reject理由
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Gate A — 答え禁止
# ---------------------------------------------------------------------------

# これらのキーが1つでもあれば即reject
BANNED_KEYS_EXACT = {
    "answer", "final", "final_answer", "solution",
    "correct", "correct_answer", "correct_option",
    "result", "output", "response",
}

# キー名に含まれていても弾く（部分一致）
BANNED_KEY_SUBSTRINGS = ["answer", "solution", "correct", "result"]


def _has_banned_key(obj: Any, depth: int = 0) -> str | None:
    """再帰的にネストされたdictを検索する。"""
    if depth > 5:
        return None
    if isinstance(obj, dict):
        for k in obj:
            k_lower = k.lower()
            if k_lower in BANNED_KEYS_EXACT:
                return k
            for sub in BANNED_KEY_SUBSTRINGS:
                if sub in k_lower and k_lower not in ("constraints", "candidate_programs"):
                    return k
            result = _has_banned_key(obj[k], depth + 1)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _has_banned_key(item, depth + 1)
            if result:
                return result
    return None


def gate_a_answer_ban(data: dict) -> str | None:
    """
    禁止フィールドが1つでもあれば reject理由を返す。
    通過なら None。
    """
    bad_key = _has_banned_key(data)
    if bad_key:
        return f"banned_key='{bad_key}'"
    return None


# ---------------------------------------------------------------------------
# Gate B — JSONスキーマ（必須フィールド）
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL = ["task", "domain_hint", "entities", "candidate_programs", "verify_spec"]

VALID_TASKS = {"mcq", "short_answer"}

ENTITY_REQUIRED = {"name", "value", "type"}

CANDIDATE_REQUIRED = {"piece", "inputs", "expected_form"}


def gate_b_schema(data: dict) -> str | None:
    """
    必須フィールドが揃っているか確認する。
    通過なら None、問題があれば reject理由。
    """
    # top-level
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            return f"missing_field='{key}'"

    # task
    if data["task"] not in VALID_TASKS:
        return f"invalid_task='{data['task']}'"

    # entities (空でもOKだがフォーマットはチェック)
    entities = data["entities"]
    if not isinstance(entities, list):
        return "entities_not_list"
    for i, e in enumerate(entities):
        if not isinstance(e, dict):
            return f"entity[{i}]_not_dict"
        missing = ENTITY_REQUIRED - set(e.keys())
        if missing:
            return f"entity[{i}]_missing={missing}"
        # ゴミ値チェック: value が記号のみ
        val = str(e.get("value", "")).strip()
        if val in ("", "**", ")", "(", "**2", "None"):
            return f"entity[{i}]_garbage_value='{val}'"

    # candidate_programs: 空はreject
    cands = data["candidate_programs"]
    if not isinstance(cands, list) or len(cands) == 0:
        return "candidate_programs_empty"
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            return f"candidate[{i}]_not_dict"
        missing = CANDIDATE_REQUIRED - set(c.keys())
        if missing:
            return f"candidate[{i}]_missing={missing}"

    # verify_spec: 存在確認（実質性はGateC）
    if not isinstance(data["verify_spec"], dict):
        return "verify_spec_not_dict"

    return None


# ---------------------------------------------------------------------------
# Gate C — verify_spec の実質性チェック（vacuous proved 根絶）
# ---------------------------------------------------------------------------

# oracle がこれだけなら vacuous
VACUOUS_ORACLES = {
    "", "True", "true", "None", "none",
    "isinstance(x, int)", "isinstance(x, str)",
    "type(x) == int", "type(x) == str",
    "pass", "...",
}

# oracle に含まれていなければならないもの（最低1つ）
# 具体的な演算か変数参照が必要
ORACLE_MUST_CONTAIN_PATTERN = re.compile(
    r"[a-zA-Z_]\w*\s*[\(\+\-\*\/\%\=\<\>]|"  # 関数呼び出し or 演算子
    r"math\.|sympy\.|numpy\.|scipy\."           # ライブラリ参照
)


def gate_c_verify_substance(data: dict) -> str | None:
    """
    verify_spec が実質的かどうかを確認する。
    vacuous spec（空・型チェックのみ・制約ゼロ）は全部reject。
    通過なら None。
    """
    spec = data.get("verify_spec", {})

    oracle = str(spec.get("oracle", "")).strip()
    constraints = spec.get("constraints", [])

    # oracle が空 or 自明
    if oracle in VACUOUS_ORACLES:
        return f"vacuous_oracle='{oracle}'"

    # oracle に演算/関数参照がなければ vacuous
    if not ORACLE_MUST_CONTAIN_PATTERN.search(oracle):
        return f"oracle_no_operation='{oracle}'"

    # constraints が完全に空
    if not isinstance(constraints, list) or len(constraints) == 0:
        return "constraints_empty"

    # constraints の中身がすべて型チェックだけなら vacuous
    type_check_only = re.compile(r"^isinstance\(|^type\(")
    non_trivial = [c for c in constraints if not type_check_only.match(str(c).strip())]
    if len(non_trivial) == 0:
        return "constraints_all_type_checks"

    return None


# ---------------------------------------------------------------------------
# 統合: 3ゲート一発チェック
# ---------------------------------------------------------------------------

def run_all_gates(data: dict) -> str | None:
    """
    GateA → GateB → GateC の順に実行。
    最初のreject理由を返す。全通過なら None。
    """
    r = gate_a_answer_ban(data)
    if r:
        return f"gate_a:{r}"
    r = gate_b_schema(data)
    if r:
        return f"gate_b:{r}"
    r = gate_c_verify_substance(data)
    if r:
        return f"gate_c:{r}"
    return None
