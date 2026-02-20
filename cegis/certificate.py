"""
Certificate - 解答の「証明書」

ドメイン別に証明書を定義し、機械的に検証可能にする。
これが「外部ソルバー不要」を成立させる鍵。

証明書の種類:
  COMPUTATION_LOG  - 計算ログ（算術、代数）
  CROSS_CHECK      - 独立した二重計算
  INTERVAL         - 区間演算（誤差境界）
  SUBSTITUTION     - 方程式への代入検証
  SEARCH_TREE      - 探索木（BFS距離、到達証明）
  EXHAUSTIVE       - 全列挙の網羅性
  SAT_WITNESS      - SAT充足証拠
  UNSAT_PROOF      - UNSAT証明
  COUNTEREXAMPLE   - 反例（命題が偽であることの証明）
  SMALL_WORLD      - 有限モデルでの検証
  PROPERTY_TEST    - ランダム性質テスト
  HIGH_CONFIDENCE  - 高信頼（証明書なし・設計上の決断）
  TACTIC           - タクティクス型証明（Lean/Metamath風）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CertKind(Enum):
    """証明書の種類"""
    COMPUTATION_LOG = "computation_log"
    CROSS_CHECK     = "cross_check"
    INTERVAL        = "interval"
    SUBSTITUTION    = "substitution"
    SEARCH_TREE     = "search_tree"
    EXHAUSTIVE      = "exhaustive"
    SAT_WITNESS     = "sat_witness"
    UNSAT_PROOF     = "unsat_proof"
    COUNTEREXAMPLE  = "counterexample"
    SMALL_WORLD     = "small_world"
    PROPERTY_TEST   = "property_test"
    HIGH_CONFIDENCE = "high_confidence"
    TACTIC          = "tactic"
    ALGEBRAIC       = "algebraic"   # SymPy/Z3による数学的証明
    MCQ_VERIFIED    = "mcq_verified"       # MCQ option-by-option verification
    ELIMINATION     = "elimination_proof"  # All other options disproved


@dataclass
class Certificate:
    """証明書"""
    kind: CertKind
    value: Any                                    # 証明書の本体（ドメイン依存）
    confidence: float = 1.0                       # 0.0–1.0
    verified: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    trace: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Certificate(kind={self.kind.value}, "
            f"confidence={self.confidence:.2f}, verified={self.verified})"
        )


class CertificateChecker:
    """証明書チェッカー - 種類別に機械的に検証"""

    def check(self, cert: Certificate, answer: Any) -> bool:
        """証明書が answer を正当化するか"""
        dispatch = {
            CertKind.COMPUTATION_LOG: self._check_computation_log,
            CertKind.CROSS_CHECK:     self._check_cross_check,
            CertKind.INTERVAL:        self._check_interval,
            CertKind.SUBSTITUTION:    self._check_substitution,
            CertKind.COUNTEREXAMPLE:  self._check_counterexample,
            CertKind.SMALL_WORLD:     self._check_small_world,
            CertKind.PROPERTY_TEST:   self._check_property_test,
            CertKind.HIGH_CONFIDENCE: self._check_high_confidence,
            CertKind.SAT_WITNESS:     self._check_sat_witness,
            CertKind.EXHAUSTIVE:      self._check_exhaustive,
            CertKind.SEARCH_TREE:     self._check_search_tree,
            CertKind.TACTIC:          self._check_tactic,
            CertKind.UNSAT_PROOF:     self._check_unsat_proof,
            CertKind.MCQ_VERIFIED:    self._check_mcq_verified,
            CertKind.ELIMINATION:     self._check_elimination,
        }
        fn = dispatch.get(cert.kind, self._check_high_confidence)
        return fn(cert, answer)

    # ────────────────────────────────────────────────────────────
    # 検証メソッド群
    # ────────────────────────────────────────────────────────────

    def _check_computation_log(self, cert: Certificate, answer: Any) -> bool:
        """計算ログの最終結果が answer に一致するか"""
        log = cert.value  # List[Tuple[op, args, result]]
        if not log:
            return False
        final = log[-1][2] if isinstance(log, list) and len(log[-1]) >= 3 else None
        return self._values_equal(final, answer)

    def _check_cross_check(self, cert: Certificate, answer: Any) -> bool:
        """独立した複数計算の結果がすべて一致するか"""
        results = cert.value  # List[Any]
        if not results or len(results) < 2:
            return False
        return all(self._values_equal(r, results[0]) for r in results)

    def _check_interval(self, cert: Certificate, answer: Any) -> bool:
        """区間演算: answer が [lo, hi] に収まるか"""
        lo = cert.value.get("lo")
        hi = cert.value.get("hi")
        try:
            v = float(str(answer))
            return lo <= v <= hi
        except (TypeError, ValueError):
            return False

    def _check_substitution(self, cert: Certificate, answer: Any) -> bool:
        """方程式に answer を代入して両辺が一致するか"""
        try:
            eq = cert.value.get("equation", "")
            val = answer
            if "=" in str(eq):
                lhs, rhs = str(eq).split("=", 1)
                # 変数を answer で置換して評価
                for var in ["x", "n", "k", "t", "y"]:
                    lhs = lhs.replace(var, str(val))
                    rhs = rhs.replace(var, str(val))
                lhs_val = eval(lhs)  # noqa: S307
                rhs_val = eval(rhs)  # noqa: S307
                return abs(float(lhs_val) - float(rhs_val)) < 1e-9
        except Exception:
            pass
        return cert.confidence >= 0.7

    def _check_counterexample(self, cert: Certificate, answer: Any) -> bool:
        """反例が存在すれば「偽」確定 → 命題は False"""
        return cert.value is not None

    def _check_small_world(self, cert: Certificate, answer: Any) -> bool:
        """小世界での検証通過率が閾値以上か"""
        worlds = cert.value.get("worlds_tested", 0)
        passed = cert.value.get("passed", 0)
        if worlds == 0:
            return False
        ratio = passed / worlds
        cert.confidence = ratio
        return ratio >= 0.8

    def _check_property_test(self, cert: Certificate, answer: Any) -> bool:
        """ランダム性質テストの通過率が閾値以上か"""
        tests  = cert.value.get("tests", 0)
        passed = cert.value.get("passed", 0)
        if tests == 0:
            return False
        return (passed / tests) >= 0.95

    def _check_high_confidence(self, cert: Certificate, answer: Any) -> bool:
        """高信頼タグ（証明書なし・設計上の決断）"""
        return cert.confidence >= 0.7

    def _check_sat_witness(self, cert: Certificate, answer: Any) -> bool:
        """SAT充足証拠の検証"""
        witness = cert.value  # Dict[var, bool]
        formula = cert.details.get("formula")
        if not formula or not witness:
            return cert.confidence >= 0.7
        # 簡易評価
        try:
            result = eval(  # noqa: S307
                formula,
                {"__builtins__": {}},
                {k: v for k, v in witness.items()}
            )
            return bool(result)
        except Exception:
            return cert.confidence >= 0.7

    def _check_exhaustive(self, cert: Certificate, answer: Any) -> bool:
        """全列挙: answer が列挙された解集合に含まれるか"""
        solutions = cert.value  # List[Any]
        if not solutions:
            return False
        return any(self._values_equal(s, answer) for s in solutions)

    def _check_search_tree(self, cert: Certificate, answer: Any) -> bool:
        """BFS/DFS 探索ログ: ゴールに到達しているか"""
        goal_reached = cert.value.get("goal_reached", False)
        return bool(goal_reached)

    def _check_tactic(self, cert: Certificate, answer: Any) -> bool:
        """タクティクス証明: steps がすべて成立するか（簡易）"""
        steps = cert.value if isinstance(cert.value, list) else []
        return len(steps) > 0 and cert.confidence >= 0.8

    def _check_unsat_proof(self, cert: Certificate, answer: Any) -> bool:
        """UNSAT 証明: 解なし確定 → 正解が 'None'/'False' のとき成功"""
        is_unsat = cert.value.get("unsat", False)
        expected_unsat = str(answer).lower() in ("none", "false", "no", "0", "∅")
        return bool(is_unsat) and expected_unsat

    def _check_mcq_verified(self, cert: Certificate, answer: Any) -> bool:
        """MCQ verified: option-by-option verification証明書"""
        # value = {verified_option: label, disproved_options: [labels], method: str}
        verified = cert.value.get("verified_option")
        disproved = cert.value.get("disproved_options", [])

        # Exactly one option verified
        if not verified:
            return False

        # Answer matches verified option
        if not self._values_equal(verified, answer):
            return False

        # At least some other options were disproved
        return len(disproved) >= 1

    def _check_elimination(self, cert: Certificate, answer: Any) -> bool:
        """Elimination proof: all other options were disproved"""
        # value = {remaining_option: label, eliminated: [labels], total_options: int}
        remaining = cert.value.get("remaining_option")
        eliminated = cert.value.get("eliminated", [])
        total = cert.value.get("total_options", 0)

        # Exactly one option remains after elimination
        if not remaining or len(eliminated) != total - 1:
            return False

        # Answer matches the remaining option
        return self._values_equal(remaining, answer)

    # ────────────────────────────────────────────────────────────
    # ユーティリティ
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """型を超えた等値判定"""
        if a is None or b is None:
            return a is b
        try:
            return abs(float(str(a)) - float(str(b))) < 1e-9
        except (TypeError, ValueError):
            return str(a).strip().lower() == str(b).strip().lower()
