"""
Enum Verifier — 有限列挙（small-world）検証器

小さな世界でサンプリングして候補を確認する。
「立体十字シミュレーション（小世界）」思想の実装。

Z3 と組み合わせて使う：
  1. 列挙で候補を生成
  2. Z3 で各候補の制約充足を確認
  または
  1. 列挙で反例を探す
  2. 見つかったら FAIL + 反例を返す
"""

from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from verifiers.api import BaseVerifier, Verdict, VerdictStatus, VerifySpec


class EnumVerifier(BaseVerifier):
    """
    有限列挙検証器。

    worldgen_params で定義された範囲でサンプルを生成し、
    候補値の正しさを抽出した条件関数で確認する。
    """
    name = "enum"

    def can_handle(self, spec: VerifySpec) -> bool:
        return spec.kind in ("enumeration", "cross_check", "numeric")

    def verify(self, candidate_value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        Worldgen params でサンプルを生成し、候補値と照合する。
        """
        wp = spec.worldgen_params or {}

        # worldgen に "cross_check_fn" が指定されている場合: 独立計算で照合
        fn_str = wp.get("cross_check_fn")
        if fn_str:
            return self._cross_check_fn(candidate_value, fn_str, wp, context)

        # 数値範囲検証
        if spec.range:
            try:
                fv = float(candidate_value)
                lo = spec.range.get("lo", -1e30)
                hi = spec.range.get("hi", 1e30)
                if lo <= fv <= hi:
                    return Verdict.pass_(self.name, witness={"in_range": fv})
                else:
                    return Verdict.fail(self.name, counterexample={"value": fv, "lo": lo, "hi": hi})
            except Exception as e:
                return Verdict.unknown(self.name, reason=f"range_error:{e}")

        return Verdict.unknown(self.name, reason="no_check_strategy")

    def _cross_check_fn(
        self,
        candidate_value: Any,
        fn_str: str,
        worldgen_params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Verdict:
        """
        独立計算関数（fn_str）で候補値を再計算して照合。

        fn_str は安全な Python 式（例: "math.factorial(n)"）。
        worldgen_params から変数値を取得する。
        """
        import math as _math

        # 変数値の取得（worldgen params または context entities から）
        var_values = {}
        entities = context.get("entities", [])
        for e in entities:
            name = e.get("name")
            value = e.get("value")
            if name and value is not None:
                var_values[name] = value

        # worldgen_params の固定値で上書き
        for k, v in worldgen_params.items():
            if k != "cross_check_fn" and not isinstance(v, dict):
                var_values[k] = v

        # 安全な評価環境
        safe_env = {
            "math": _math,
            "factorial": _math.factorial,
            "gcd": _math.gcd,
            "sqrt": _math.sqrt,
            "abs": abs,
            "pow": pow,
            "min": min,
            "max": max,
            **var_values,
        }

        try:
            expected = eval(fn_str, {"__builtins__": {}}, safe_env)
            if expected is None:
                return Verdict.unknown(self.name, reason="cross_check_fn_returned_none")

            # 照合
            tol = 1e-9
            try:
                if abs(float(candidate_value) - float(expected)) <= tol:
                    return Verdict.pass_(
                        self.name,
                        witness={"expected": expected, "got": candidate_value, "fn": fn_str}
                    )
                else:
                    return Verdict.fail(
                        self.name,
                        counterexample={
                            "expected": expected,
                            "got": candidate_value,
                            "fn": fn_str,
                            "vars": var_values
                        },
                        details=[f"Expected {expected}, got {candidate_value}"]
                    )
            except (TypeError, ValueError):
                if str(candidate_value) == str(expected):
                    return Verdict.pass_(self.name)
                return Verdict.fail(
                    self.name,
                    counterexample={"expected": str(expected), "got": str(candidate_value)}
                )

        except Exception as e:
            return Verdict.unknown(self.name, reason=f"cross_check_fn_error:{e}")


# ─────────────────────────────────────────────────────────────────────
# Small-World Sampler
# ─────────────────────────────────────────────────────────────────────

class SmallWorldSampler:
    """
    Verantyx の「立体十字シミュレーション（小世界）」補助クラス。

    worldgen spec から small-world サンプルを生成し、
    候補関数の反例を探す。
    """

    @staticmethod
    def sample_integers(
        var_spec: Dict[str, Any],
        n_samples: int = 20,
        seed: int = 42
    ) -> List[Dict[str, int]]:
        """
        整数変数のサンプルを生成する。

        var_spec 例:
          {"n": {"min": 0, "max": 12}, "r": {"min": 0, "max": "n"}}
        """
        random.seed(seed)
        samples = []

        for _ in range(n_samples):
            sample: Dict[str, int] = {}
            for var_name, var_info in var_spec.items():
                if not isinstance(var_info, dict):
                    continue
                lo = int(var_info.get("min", 0))
                # "max": "n" のような相互参照に対応
                hi_raw = var_info.get("max", 10)
                if isinstance(hi_raw, str) and hi_raw in sample:
                    hi = sample[hi_raw]
                else:
                    try:
                        hi = int(hi_raw)
                    except (TypeError, ValueError):
                        hi = 10
                if lo > hi:
                    lo, hi = hi, lo
                sample[var_name] = random.randint(lo, hi)
            if sample:
                samples.append(sample)

        return samples

    @staticmethod
    def find_counterexample(
        candidate_fn: Callable[..., Any],
        verify_fn: Callable[..., bool],
        var_spec: Dict[str, Any],
        n_samples: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        候補関数の反例をサンプリングで探す。

        Args:
            candidate_fn: 候補の計算関数
            verify_fn: 正しさの検証関数（True = 正しい）
            var_spec: 変数仕様
            n_samples: サンプル数

        Returns:
            反例辞書 or None（反例なし）
        """
        samples = SmallWorldSampler.sample_integers(var_spec, n_samples)
        for sample in samples:
            try:
                result = candidate_fn(**sample)
                is_correct = verify_fn(result, **sample)
                if not is_correct:
                    return {"input": sample, "got": result}
            except Exception:
                pass
        return None
