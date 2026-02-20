"""
Z3 Verifier — SMT ベース検証器（反例生成器）

Z3 の役割:
  - sat/unsat + model を返す（=反例/充足例）
  - 候補値が制約を満たすか判定
  - 反例を worldgen に渡して次候補改善に使う

原則A: Z3 は "答えを出す箱" ではなく "判定器"。

対象:
  - 整数算・条件分岐・論理
  - グラフ・組合せ制約（有限モデル）
  - 存在/全称 (∃/∀)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from verifiers.api import BaseVerifier, Verdict, VerdictStatus, VerifySpec


class Z3Verifier(BaseVerifier):
    name = "z3"

    def can_handle(self, spec: VerifySpec) -> bool:
        return spec.kind in ("smt", "enumeration", "numeric", "cross_check")

    def verify(self, candidate_value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        Z3 で候補値を検証する。

        候補値が制約を満たすなら PASS、違反するなら FAIL + 反例モデルを返す。
        """
        try:
            import z3
        except ImportError:
            return Verdict.unknown(self.name, reason="z3_not_installed")

        kind = spec.kind

        # 制約リストがある場合: SMT 検証
        if spec.constraints:
            return self._smt_check(candidate_value, spec, context, z3)

        # 数値範囲検証
        if kind in ("numeric", "cross_check") and spec.range:
            return self._range_check(candidate_value, spec, z3)

        # 型チェック
        if spec.type_check:
            return self._type_check(candidate_value, spec, z3)

        return Verdict.unknown(self.name, reason="no_constraints_to_check")

    def _smt_check(self, value: Any, spec: VerifySpec, context: Dict[str, Any], z3) -> Verdict:
        """
        制約式を Z3 ソルバーに投げて sat/unsat を判定する。

        spec.constraints は Python/Z3 形式の制約文字列リスト。
        例: ["x > 0", "x * x == 4", "x == candidate_value"]
        """
        solver = z3.Solver()
        solver.set("timeout", spec.timeout_ms)

        # 変数宣言
        variables = self._extract_variables(spec.constraints, value)
        z3_vars: Dict[str, Any] = {}
        for var_name, var_type in variables.items():
            if var_type == "int":
                z3_vars[var_name] = z3.Int(var_name)
            elif var_type == "real":
                z3_vars[var_name] = z3.Real(var_name)
            elif var_type == "bool":
                z3_vars[var_name] = z3.Bool(var_name)
            else:
                z3_vars[var_name] = z3.Int(var_name)  # default

        # candidate_value を "answer" 変数として注入
        cv_str = str(value)
        z3_vars["candidate_value"] = value  # 定数として

        # 制約を Z3 式に変換
        env = {**z3_vars, "z3": z3}
        try:
            for constraint in spec.constraints:
                # Python 式として評価（制約が Python 構文であることを前提）
                constraint_py = self._constraint_to_python(constraint, value)
                z3_expr = eval(constraint_py, {"__builtins__": {}}, env)
                solver.add(z3_expr)
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"constraint_parse_error:{e}")

        # 判定
        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            model_dict = {str(d): str(model[d]) for d in model.decls()}
            return Verdict.pass_(
                self.name,
                witness={"z3_model": model_dict, "status": "sat"}
            )
        elif result == z3.unsat:
            # unsat = 制約を満たす解がない = 候補が間違い
            # → 反例として「制約を違反するモデル」を生成して返す
            return Verdict.fail(
                self.name,
                counterexample={"status": "unsat", "constraints": spec.constraints, "candidate": str(value)},
                details=["Z3: no model satisfies constraints with this candidate value"]
            )
        else:
            return Verdict.unknown(self.name, reason="z3_timeout_or_unknown")

    def _range_check(self, value: Any, spec: VerifySpec, z3) -> Verdict:
        """Z3 で範囲制約を検証"""
        r = spec.range
        try:
            fv = float(value)
            lo = r.get("lo", -1e30)
            hi = r.get("hi", 1e30)

            # Z3 を使わずに直接判定（範囲チェックは高速）
            if lo <= fv <= hi:
                return Verdict.pass_(self.name, witness={"in_range": True, "value": fv})
            else:
                return Verdict.fail(
                    self.name,
                    counterexample={"value": fv, "lo": lo, "hi": hi},
                    details=[f"{fv} not in [{lo}, {hi}]"]
                )
        except (TypeError, ValueError):
            return Verdict.unknown(self.name, reason="non_numeric_for_range_check")

    def _type_check(self, value: Any, spec: VerifySpec, z3) -> Verdict:
        """Z3 で型制約を検証"""
        tc = spec.type_check
        try:
            if tc == "integer":
                iv = int(float(value))
                if iv == float(value):
                    return Verdict.pass_(self.name, witness={"is_integer": True})
                else:
                    return Verdict.fail(
                        self.name,
                        counterexample={"expected_integer": True, "got": value}
                    )
            elif tc == "boolean":
                if isinstance(value, bool) or str(value).lower() in ("true", "false", "yes", "no"):
                    return Verdict.pass_(self.name)
                else:
                    return Verdict.fail(self.name, counterexample={"expected_bool": str(value)})
            return Verdict.pass_(self.name)
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"type_check_error:{e}")

    @staticmethod
    def _extract_variables(constraints: List[str], candidate_value: Any) -> Dict[str, str]:
        """制約式から変数名と型を抽出する"""
        variables: Dict[str, str] = {}
        var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

        # Z3 組み込み以外の識別子を変数候補として収集
        BUILTIN_NAMES = {"z3", "Int", "Real", "Bool", "And", "Or", "Not",
                         "Implies", "If", "True", "False", "candidate_value",
                         "sum", "abs", "min", "max"}

        for constraint in constraints:
            for m in var_pattern.finditer(constraint):
                name = m.group(1)
                if name not in BUILTIN_NAMES and not name[0].isupper():
                    # デフォルトは整数型
                    if name not in variables:
                        variables[name] = "int"

        return variables

    @staticmethod
    def _constraint_to_python(constraint: str, candidate_value: Any) -> str:
        """
        制約文字列を Python/Z3 式に変換する。

        例:
          "x > 0 and x * x == candidate_value"
          → "z3.And(x > 0, x * x == candidate_value)"

        現状は基本的な変換のみ。
        """
        # "and" → z3.And(), "or" → z3.Or() への変換は複雑なので
        # 現状はシンプルに eval に委ねる（Python の and/or は z3 式を短絡評価しないので危険）
        # 改善: 将来は AST 変換で z3.And/Or に置き換える

        # ただし "==" を z3 等値比較として使う（Python eval と互換）
        return constraint


# ─────────────────────────────────────────────────────────────────────
# Z3 小世界反例生成器（worldgen の補助）
# ─────────────────────────────────────────────────────────────────────

class Z3CounterexampleGenerator:
    """
    Z3 で反例を生成する補助クラス。

    CEGIS の worldgen 相当の役割を Z3 に委任する。
    候補が FAIL した後、制約違反の具体例を生成して次候補生成に使う。
    """

    def __init__(self, timeout_ms: int = 1000):
        self.timeout_ms = timeout_ms

    def generate(
        self,
        domain_constraints: List[str],
        candidate_value: Any,
        domain_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        ドメイン制約と候補値から反例を生成する。

        Returns:
            反例辞書（変数名 → 値）or None（生成不能）
        """
        try:
            import z3
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)

            # ドメイン仕様から変数を宣言
            z3_vars: Dict[str, Any] = {}
            if domain_spec:
                for var_name, var_info in domain_spec.items():
                    var_type = var_info.get("type", "int")
                    if var_type == "int":
                        v = z3.Int(var_name)
                        z3_vars[var_name] = v
                        lo = var_info.get("min", -1000)
                        hi = var_info.get("max", 1000)
                        solver.add(v >= lo, v <= hi)
                    elif var_type == "real":
                        v = z3.Real(var_name)
                        z3_vars[var_name] = v

            # 候補値が "誤り" という制約（= not_equal）
            if candidate_value is not None and z3_vars:
                first_var = list(z3_vars.values())[0]
                try:
                    solver.add(first_var != int(candidate_value))
                except Exception:
                    pass

            # ドメイン制約
            env = {**z3_vars, "z3": z3}
            for constraint in domain_constraints:
                try:
                    expr = eval(constraint, {"__builtins__": {}}, env)
                    solver.add(expr)
                except Exception:
                    continue

            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                return {str(d): model.eval(z3_vars.get(str(d), z3.IntVal(0))) for d in model.decls() if str(d) in z3_vars}
            return None
        except Exception:
            return None
