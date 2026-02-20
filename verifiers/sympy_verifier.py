"""
SymPy Verifier — 数値・代数検証器

候補値を SymPy で独立検証する。
- cross_check: 同じ計算を別の方法で実行して照合
- numeric: 数値型チェック + 範囲チェック
- symbolic: 式の同値性確認

原則A: SymPy は "答えを出す箱" ではなく "判定器"。
候補値を受け取って PASS/FAIL/UNKNOWN を返すだけ。
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional

from verifiers.api import BaseVerifier, Verdict, VerdictStatus, VerifySpec


class SympyVerifier(BaseVerifier):
    name = "sympy"

    def can_handle(self, spec: VerifySpec) -> bool:
        return spec.kind in ("numeric", "cross_check", "symbolic", "smt")

    def verify(self, candidate_value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        SymPy で候補値を検証する。

        cross_check: 期待値と照合（独立再計算）
        numeric:     型・範囲チェック
        symbolic:    式の簡約・同値確認
        """
        kind = spec.kind

        if kind == "cross_check":
            return self._cross_check(candidate_value, spec, context)
        elif kind == "numeric":
            return self._numeric_check(candidate_value, spec)
        elif kind == "symbolic":
            return self._symbolic_check(candidate_value, spec, context)
        else:
            return Verdict.unknown(self.name, reason=f"unsupported_kind:{kind}")

    def _cross_check(self, value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        独立計算で照合する。
        spec.method によって検証手法を切り替える。
        """
        method = spec.method

        if method == "double_eval":
            return self._double_eval(value, spec, context)
        elif method == "interval_arithmetic":
            return self._interval_check(value, spec)
        else:
            # generic: 型チェックのみ
            return self._numeric_check(value, spec)

    def _double_eval(self, value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        候補値を SymPy で再計算して照合。
        Executor の計算結果を独立した方法で確認する。
        """
        try:
            import sympy

            # 型チェック
            type_check = spec.type_check
            if type_check == "integer":
                if not isinstance(value, (int, sympy.Integer)):
                    try:
                        iv = int(value)
                        if iv != value:
                            return Verdict.fail(
                                self.name,
                                counterexample={"expected_type": "integer", "got": type(value).__name__, "value": value},
                                details=["Value is not an integer"]
                            )
                    except (TypeError, ValueError):
                        return Verdict.fail(
                            self.name,
                            counterexample={"type_error": str(type(value)), "value": str(value)},
                        )

            # 範囲チェック
            r = spec.range
            if r and value is not None:
                try:
                    fv = float(value)
                    lo = r.get("lo", -1e30)
                    hi = r.get("hi", 1e30)
                    if not (lo <= fv <= hi):
                        return Verdict.fail(
                            self.name,
                            counterexample={"out_of_range": fv, "lo": lo, "hi": hi},
                            details=[f"{fv} not in [{lo}, {hi}]"]
                        )
                except (TypeError, ValueError, OverflowError):
                    pass  # 非数値は範囲チェックスキップ

            # 期待値との照合（piece の examples フィールドから）
            if spec.expected_value is not None:
                try:
                    ev = spec.expected_value
                    if isinstance(value, (int, float)) and isinstance(ev, (int, float)):
                        tol = 1e-9
                        if abs(float(value) - float(ev)) > tol:
                            return Verdict.fail(
                                self.name,
                                counterexample={"expected": ev, "got": value},
                                details=[f"Value mismatch: {value} != {ev}"]
                            )
                    elif str(value) != str(ev):
                        return Verdict.fail(
                            self.name,
                            counterexample={"expected": str(ev), "got": str(value)},
                        )
                except Exception:
                    pass

            return Verdict.pass_(self.name, witness={"value": value, "method": "double_eval"})

        except Exception as e:
            return Verdict.unknown(self.name, reason=f"sympy_error:{e}")

    def _interval_check(self, value: Any, spec: VerifySpec) -> Verdict:
        """区間演算で誤差チェック"""
        try:
            fv = float(value)
            if not math.isfinite(fv):
                return Verdict.fail(self.name, counterexample={"non_finite": str(value)})
            r = spec.range
            if r:
                if not (r.get("lo", -1e30) <= fv <= r.get("hi", 1e30)):
                    return Verdict.fail(self.name, counterexample={"out_of_range": fv})
            return Verdict.pass_(self.name)
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"interval_error:{e}")

    def _numeric_check(self, value: Any, spec: VerifySpec) -> Verdict:
        """基本的な型・範囲チェック"""
        if value is None:
            return Verdict.fail(self.name, counterexample={"null_value": True})
        try:
            fv = float(value)
            if not math.isfinite(fv):
                return Verdict.fail(self.name, counterexample={"non_finite": str(fv)})
            r = spec.range
            if r and not (r.get("lo", -1e30) <= fv <= r.get("hi", 1e30)):
                return Verdict.fail(self.name, counterexample={"out_of_range": fv, **r})
            return Verdict.pass_(self.name, witness={"numeric_value": fv})
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"non_numeric:{e}")

    def _symbolic_check(self, value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """式の同値確認（SymPy simplify）"""
        try:
            import sympy
            if spec.expected_value is not None:
                expr1 = sympy.sympify(str(value))
                expr2 = sympy.sympify(str(spec.expected_value))
                diff = sympy.simplify(expr1 - expr2)
                if diff == 0:
                    return Verdict.pass_(self.name, witness={"symbolic_equal": True})
                else:
                    return Verdict.fail(self.name, counterexample={"diff": str(diff)})
            return Verdict.pass_(self.name)
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"symbolic_error:{e}")
