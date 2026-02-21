"""
CEGIS Loop - CounterExample Guided Inductive Synthesis

外部ソルバー不要でPhD級問題に対応するための核心ループ。

8ステップの推論フロー:
  1. Decompose   - IR生成（外部呼び出し済みで渡される）
  2. Plan Search - 候補リスト初期化
  3. Retrieve    - Cross DB からピース選択（外部呼び出し済み）
  4. Build Micro-World - 有限モデル仕様を作成
  5. Simulate    - 小世界で候補をフィルタ（反例探索）
  6. Refine      - 反例から制約を一般化して候補再生成
  7. Certify     - 生き残り候補に証明書を生成・検証
  8. Render      - Grammar Glue で文字列化（外部呼び出し済み）

設計哲学:
  「候補=文字列」を全面禁止。候補=構造体+証明書。
  文字列は Grammar Glue 層だけが出す。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


# ─── ChatGPT 設計: 3値 CheckStatus ───────────────────────────────────────────
class CheckStatus(Enum):
    PROVED      = auto()   # 反例なし・検証完了
    DISPROVED   = auto()   # 反例あり
    INCONCLUSIVE = auto()  # worldgen失敗 / schema不足 / 実行不能 (→ reject)


@dataclass
class WorldGenResult:
    ok: bool
    world: Any
    error: str | None = None


def safe_worldgen(world_gen, domain: str, params: dict) -> "WorldGenResult":
    """worldgen 失敗を例外ごと握りつぶさず INCONCLUSIVE に変換"""
    try:
        worlds = world_gen.generate(domain, params)
        if not worlds:
            return WorldGenResult(False, None, "worldgen_returned_empty")
        return WorldGenResult(True, worlds, None)
    except Exception as e:
        return WorldGenResult(False, None, f"worldgen_exception:{type(e).__name__}:{e}")


def candidate_sanity(domain: str, value: Any) -> bool:
    """候補 answer の型・範囲チェック（trivial pass 防波堤）"""
    if domain == "multiple_choice" or (isinstance(value, str) and value in list("ABCDE")):
        return isinstance(value, str) and value in ["A", "B", "C", "D", "E"]
    if isinstance(value, (int, float)):
        try:
            return abs(float(value)) < 1e12  # 兆を超える数値は疑わしい
        except Exception:
            return False
    return True  # 文字列などは通す
# ─────────────────────────────────────────────────────────────────────────────

from .certificate import Certificate, CertKind, CertificateChecker
from .worldgen import FiniteModel, WorldGenerator

# Verifier API (原則A/B/C)
try:
    from verifiers.api import verify as _verifier_verify, VerifySpec, VerdictStatus
    _VERIFIER_AVAILABLE = True
except ImportError:
    _VERIFIER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# データ構造
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """
    候補（解・証明計画）

    value: 解の値（数値、式、ラベル等）
    construction: 使ったピースのID列（トレーサビリティ）
    confidence: 事前推定信頼度
    constraints: この候補が満たすべき制約のリスト
    """
    value: Any
    construction: List[str] = field(default_factory=list)
    confidence: float = 0.5
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Candidate(value={self.value!r}, conf={self.confidence:.2f})"


@dataclass
class WorldSpec:
    """有限モデル生成仕様（Step 4 の出力）"""
    domain: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class CEGISResult:
    """CEGIS ループの最終結果"""
    answer: Any                                  # 解（None なら失敗）
    confidence: float                            # 0.0–1.0
    certificate: Optional[Certificate]           # 証明書（あれば）
    iterations: int                              # 実行したイテレーション数
    counterexamples: List[Any] = field(default_factory=list)
    status: str = "unknown"   # "proved" | "disproved" | "high_confidence" | "timeout" | "unknown"
    trace: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CEGISResult(status={self.status}, "
            f"answer={self.answer!r}, "
            f"conf={self.confidence:.2f}, "
            f"iters={self.iterations})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ドメイン→小世界ドメインのマッピング
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_TO_WORLD: Dict[str, str] = {
    "arithmetic":               "number",
    "algebra":                  "substitution",    # 変数代入で恒等式検証
    "linear_algebra":           "matrix",
    "calculus":                 "polynomial",
    "logic_propositional":      "propositional",
    "logic_modal":              "propositional",
    "logic_first_order":        "set",
    "graph_theory":             "graph",
    "number_theory":            "modular",         # mod p 演算
    "combinatorics":            "sequence",
    "probability":              "number",
    "statistics":               "sequence",
    "modular_arithmetic":       "modular",         # mod p 演算
    "advanced_probability":     "number",
    "advanced_number_theory":   "modular",         # mod p 演算
    "advanced_combinatorics":   "sequence",
    "string":                   "sequence",
    "multiple_choice":          "number",
    "puzzle":                   "set",
    "physics":                  "number",
    "chemistry":                "number",
    "computer_science":         "graph",
    "philosophy":               "propositional",
    "group_theory":             "finite_group",    # 有限群
    "field_theory":             "finite_field",    # 有限体
    "polynomial":               "substitution",    # 多項式恒等式
    "unknown":                  "number",
}


# ─────────────────────────────────────────────────────────────────────────────
# メインクラス
# ─────────────────────────────────────────────────────────────────────────────


class CEGISLoop:
    """
    CEGIS (CounterExample Guided Inductive Synthesis) ループ

    外部ソルバー不要で PhD 級問題に対応するための核心ループ。

    使い方:
        loop = CEGISLoop(max_iter=5, max_worlds=50)
        result = loop.run(ir_dict, candidates)
        print(result.answer, result.confidence)
    """

    def __init__(
        self,
        max_iter: int = 5,
        max_worlds: int = 50,
        max_candidates: int = 10,
        time_limit_ms: float = 5000.0,
    ):
        self.max_iter = max_iter
        self.world_gen = WorldGenerator(max_worlds=max_worlds)
        self.cert_checker = CertificateChecker()
        self.max_candidates = max_candidates
        self.time_limit_ms = time_limit_ms

    # ─────────────────────────────────────────────────────────────────────
    # メインループ
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        ir_dict: Dict[str, Any],
        candidates: List[Candidate],
        candidate_generator: Optional[Callable[[List[str]], List[Candidate]]] = None,
        world_spec_override: Optional[WorldSpec] = None,
    ) -> CEGISResult:
        """
        CEGIS メインループ

        Args:
            ir_dict: IR 辞書（問題の構造）
            candidates: 初期候補リスト（Step 3 の出力）
            candidate_generator: 制約リストを受け取って新候補を生成するコールバック
            world_spec_override: 有限モデル仕様の上書き（省略時は IR から自動推定）

        Returns:
            CEGISResult
        """
        t_start = time.perf_counter()
        trace: List[str] = []
        counterexamples: List[Any] = []
        current_candidates = candidates[: self.max_candidates]

        # Step 4: 有限モデル仕様を決定
        world_spec = world_spec_override or self._infer_world_spec(ir_dict)
        trace.append(f"CEGIS start | candidates={len(current_candidates)} | world={world_spec.domain}")
        trace.append(f"  reason: {world_spec.reason}")

        for iteration in range(self.max_iter):
            # タイムリミットチェック
            elapsed = (time.perf_counter() - t_start) * 1000
            if elapsed > self.time_limit_ms:
                trace.append(f"iter={iteration}: TIME LIMIT ({elapsed:.0f}ms)")
                break

            if not current_candidates:
                trace.append(f"iter={iteration}: no candidates left")
                break

            # ─── Step 5: Simulate（小世界生成 + 反例テスト） ────────────────
            wg_result = safe_worldgen(self.world_gen, world_spec.domain, world_spec.params)
            if not wg_result.ok:
                trace.append(f"iter={iteration}: INCONCLUSIVE — worldgen failed: {wg_result.error}")
                break  # worldgen 失敗 → このループでは証明不可能
            worlds = wg_result.world
            trace.append(f"iter={iteration}: {len(worlds)} worlds generated")

            surviving: List[Candidate] = []
            for cand in current_candidates:
                # 答えの sanity check（MCQ=A-E 強制、数値=範囲チェック）
                domain = ir_dict.get("domain", "unknown")
                if not candidate_sanity(domain, cand.value):
                    trace.append(f"  ✗ cand={cand.value!r} | SANITY_FAIL (domain={domain})")
                    counterexamples.append({"sanity": "failed", "value": str(cand.value)})
                    continue
                counterex = self._find_counterexample(cand, worlds, ir_dict)
                if counterex is None:
                    surviving.append(cand)
                else:
                    counterexamples.append(counterex)
                    trace.append(
                        f"  ✗ cand={cand.value!r} | counterex={counterex}"
                    )

            trace.append(
                f"iter={iteration}: survived {len(surviving)}/{len(current_candidates)}"
            )

            # ─── Step 6: Refine（反例→制約→候補再生成） ───────────────────
            if not surviving:
                if candidate_generator:
                    new_constraints = self._generalize_counterexamples(counterexamples)
                    trace.append(f"  refine: constraints={new_constraints}")
                    current_candidates = candidate_generator(new_constraints)[: self.max_candidates]
                    counterexamples = []
                    trace.append(f"  regenerated {len(current_candidates)} candidates")
                    continue
                else:
                    trace.append(f"iter={iteration}: no generator, exhausted")
                    break

            # ─── Step 7: Certify（証明書生成・検証） ────────────────────────
            for cand in sorted(surviving, key=lambda c: -c.confidence):
                cert = self._build_certificate(cand, worlds, ir_dict)
                if cert is None:
                    continue
                if self.cert_checker.check(cert, cand.value):
                    cert.verified = True
                    status = "proved" if cert.kind != CertKind.HIGH_CONFIDENCE else "high_confidence"
                    trace.append(f"  ✓ cand={cand.value!r} | cert={cert.kind.value}")
                    return CEGISResult(
                        answer=cand.value,
                        confidence=cert.confidence,
                        certificate=cert,
                        iterations=iteration + 1,
                        counterexamples=counterexamples,
                        status=status,
                        trace=trace,
                        elapsed_ms=(time.perf_counter() - t_start) * 1000,
                    )

            # ⚠️ HIGH_CONFIDENCE fallback は廃止。
            # 証明書なしで confidence だけで通すのは trivial pass の源泉 → INCONCLUSIVE 扱い。
            # (ChatGPT 設計: INCONCLUSIVE は reject)

            current_candidates = surviving

        # ─── タイムアウト: 最善候補を返す ────────────────────────────────────
        if current_candidates:
            best = max(current_candidates, key=lambda c: c.confidence)
            trace.append(f"TIMEOUT: best={best.value!r} conf={best.confidence:.2f}")
            return CEGISResult(
                answer=best.value,
                confidence=best.confidence * 0.5,
                certificate=None,
                iterations=self.max_iter,
                counterexamples=counterexamples,
                status="timeout",
                trace=trace,
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            )

        trace.append("UNKNOWN: no candidates survived")
        return CEGISResult(
            answer=None,
            confidence=0.0,
            certificate=None,
            iterations=self.max_iter,
            counterexamples=counterexamples,
            status="unknown",
            trace=trace,
            elapsed_ms=(time.perf_counter() - t_start) * 1000,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: 有限モデル仕様の推定
    # ─────────────────────────────────────────────────────────────────────

    def _infer_world_spec(self, ir_dict: Dict[str, Any]) -> WorldSpec:
        """IR ドメインから有限モデル生成仕様を自動推定"""
        domain = ir_dict.get("domain", "unknown")
        world_domain = _DOMAIN_TO_WORLD.get(domain, "number")

        # ドメイン別のパラメータ調整
        params: Dict[str, Any] = {}
        if world_domain == "number":
            params = {"lo": -20, "hi": 20}
        elif world_domain == "propositional":
            # IR から原子命題を抽出
            entities = ir_dict.get("entities", [])
            atoms = [e.get("name") for e in entities if e.get("type") == "symbol"]
            if not atoms:
                atoms = ["p", "q", "r"]
            params = {"atoms": atoms[:5]}
        elif world_domain == "graph":
            params = {"n_range": range(3, 7)}
        elif world_domain == "matrix":
            params = {"dim": 2, "val_range": range(-2, 3)}
        elif world_domain == "sequence":
            params = {"length": 8}
        elif world_domain == "substitution":
            # IR から変数名を抽出
            entities = ir_dict.get("entities", [])
            vars_list = [e.get("name") for e in entities if e.get("type") in ["variable", "symbol"]]
            if not vars_list:
                vars_list = ["x", "y", "n", "k", "a", "b"]
            params = {"vars": vars_list[:6], "count": 30}
        elif world_domain == "finite_field":
            params = {"primes": [2, 3, 5, 7, 11, 13]}
        elif world_domain == "finite_group":
            params = {"orders": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]}
        elif world_domain == "modular":
            params = {"moduli": [2, 3, 5, 7, 11, 13, 17, 19, 23]}

        return WorldSpec(
            domain=world_domain,
            params=params,
            reason=f"IR domain={domain} → world={world_domain}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: 反例テスト
    # ─────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    # Verifier API 統合（原則A/B/C）
    # ─────────────────────────────────────────────────────────────────────

    def _build_basic_verify_spec(
        self, cand: Candidate, ir_dict: Dict[str, Any]
    ) -> Optional["VerifySpec"]:
        """
        候補の制約リストから基本的な VerifySpec を構築する。
        Piece に verify フィールドがない場合のフォールバック用。
        """
        if not _VERIFIER_AVAILABLE:
            return None

        constraints = cand.constraints or []

        # 制約 → 型/範囲パラメータに変換
        type_check = None
        range_params = None

        for c in constraints:
            if c == "integer":
                type_check = "integer"
            elif c == "positive":
                range_params = {"lo": 0, "hi": 1e15}
            elif c == "non_negative":
                range_params = {"lo": 0, "hi": 1e15}
            elif c == "negative":
                range_params = {"lo": -1e15, "hi": 0}

        # 値が数値でなければ検証スキップ
        try:
            float(str(cand.value))
        except (TypeError, ValueError):
            return None

        return VerifySpec(
            kind="numeric",
            method="double_eval",
            type_check=type_check,
            range=range_params,
            timeout_ms=500,
        )

    def _verify_with_api(
        self, cand: Candidate, ir_dict: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Verifier API（SymPy → Z3 → Enum カスケード）で候補を検証。

        Returns:
            None      → PASS（反例なし）
            dict      → FAIL（counterexample）
            "unknown" → UNKNOWN（判定不能、旧小世界フォールバックへ）
        """
        if not _VERIFIER_AVAILABLE:
            return "unknown"

        # Step 1: Piece の verify_spec があればそれを使う
        verify_spec_dict = cand.metadata.get("verify_spec")
        if verify_spec_dict and isinstance(verify_spec_dict, dict):
            try:
                spec = VerifySpec.from_piece_dict(verify_spec_dict)
            except Exception:
                spec = None
        else:
            spec = None

        # Step 2: なければ基本 spec を構築
        if spec is None:
            spec = self._build_basic_verify_spec(cand, ir_dict)

        if spec is None:
            return "unknown"

        # Step 3: Verifier カスケード呼び出し
        try:
            verdict = _verifier_verify(cand.value, spec, context=ir_dict)
            if verdict.status == VerdictStatus.PASS:
                # PASS: 候補は正しい → 反例なし
                return None
            elif verdict.status == VerdictStatus.FAIL:
                # FAIL: 候補は誤り → 反例をCEGISに返す
                ce = verdict.counterexample or {"verifier": verdict.verifier, "reason": "fail"}
                ce["_source"] = f"verifier:{verdict.verifier}"
                return ce
            else:
                # UNKNOWN: 旧小世界フォールバック
                return "unknown"
        except Exception:
            return "unknown"

    def _find_counterexample(
        self,
        cand: Candidate,
        worlds: List[FiniteModel],
        ir_dict: Dict[str, Any],
    ) -> Optional[Any]:
        """
        候補を検証して反例を返す。

        Step 1: Verifier API (SymPy → Z3 → Enum) — 数学的証明
        Step 2: 小世界テスト — フォールバック
        反例なし → None。
        """
        # ─── Verifier API を先に試す ───
        api_result = self._verify_with_api(cand, ir_dict)
        if api_result is None:
            # PASS: SymPy/Z3 で数学的に正しいと判定
            cand.metadata["verifier_pass"] = True
            return None
        elif api_result != "unknown":
            # FAIL: 反例あり
            cand.metadata["verifier_fail"] = True
            return api_result
        # UNKNOWN → 旧小世界テストにフォールバック

        # ─── 旧: 小世界テスト ───
        constraints = cand.constraints
        if not constraints:
            # 制約なしは「どんな世界でも壊れない」ではなく「検証できない」= INCONCLUSIVE 扱い
            # → 反例を返さないが verifier_pass も付かないので cert_checker が落とす
            return None

        for world in worlds[:20]:
            for constraint in constraints:
                if self._violates_constraint(cand.value, world, constraint):
                    return {
                        "world_domain": world.domain,
                        "world_size": world.size,
                        "constraint": constraint,
                        "candidate": cand.value,
                    }
        return None

    def _violates_constraint(
        self, value: Any, world: FiniteModel, constraint: str
    ) -> bool:
        """制約違反チェック（基本セット）"""
        try:
            v = float(str(value))
        except (TypeError, ValueError):
            v = None

        checks = {
            "positive":     lambda: v is not None and v <= 0,
            "negative":     lambda: v is not None and v >= 0,
            "non_negative": lambda: v is not None and v < 0,
            "integer":      lambda: v is not None and v != int(v),
            "even":         lambda: v is not None and int(v) % 2 != 0,
            "odd":          lambda: v is not None and int(v) % 2 == 0,
            "prime":        lambda: v is not None and not self._is_prime_check(int(v)),
        }
        checker = checks.get(constraint)
        if checker:
            try:
                return checker()
            except Exception:
                return False

        # 世界プロパティとのクロスチェック
        if constraint in world.properties:
            expected = world.properties[constraint]
            if isinstance(value, bool):
                return value != expected

        return False

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: 反例の一般化
    # ─────────────────────────────────────────────────────────────────────

    def _generalize_counterexamples(self, counterexamples: List[Any]) -> List[str]:
        """反例から新しい制約セットを一般化"""
        constraints: List[str] = []
        for ce in counterexamples:
            if isinstance(ce, dict):
                c = ce.get("constraint")
                if c and c not in constraints:
                    constraints.append(c)
        return constraints

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: 証明書生成
    # ─────────────────────────────────────────────────────────────────────

    def _build_certificate(
        self,
        cand: Candidate,
        worlds: List[FiniteModel],
        ir_dict: Dict[str, Any],
    ) -> Optional[Certificate]:
        """候補のための証明書を構築"""

        # ─── Verifier API で既に PASS している場合は最高位証明書 ───
        if cand.metadata.get("verifier_pass"):
            return Certificate(
                kind=CertKind.ALGEBRAIC,   # 数学的証明（最高位）
                value={"verifier": "sympy_or_z3", "worlds_tested": 0},
                confidence=min(cand.confidence * 1.2, 1.0),   # 信頼度を上げる
                details={
                    "candidate": str(cand.value),
                    "construction": cand.construction,
                    "verified_by": "Verifier API (SymPy/Z3)",
                },
            )

        passed = sum(
            1 for w in worlds
            if self._find_counterexample(cand, [w], ir_dict) is None
        )
        total = len(worlds)

        if total == 0:
            # worldgen 失敗 or 空 → INCONCLUSIVE → 証明書なし（ChatGPT ルール1）
            return None

        ratio = passed / total
        if ratio >= 0.9:
            kind = CertKind.SMALL_WORLD
        elif ratio >= 0.7:
            kind = CertKind.PROPERTY_TEST
        else:
            kind = CertKind.HIGH_CONFIDENCE

        return Certificate(
            kind=kind,
            value={"worlds_tested": total, "passed": passed, "ratio": ratio},
            confidence=ratio * cand.confidence,
            details={
                "candidate": str(cand.value),
                "construction": cand.construction,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # ユーティリティ
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_prime_check(n: int) -> bool:
        import math
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

    # ─────────────────────────────────────────────────────────────────────
    # MCQ-specific verification
    # ─────────────────────────────────────────────────────────────────────

    def verify_mcq_options(
        self,
        question_text: str,
        choices: Dict[str, str],
        ir_dict: Dict[str, Any],
    ) -> CEGISResult:
        """
        MCQ-specific verification: test each option A/B/C/D/E independently.

        Strategy:
        1. For each option, create a candidate
        2. Use CEGIS to verify or disprove each option
        3. If exactly one option is verified → return it with MCQ_VERIFIED cert
        4. If all but one are disproved → return remaining with ELIMINATION cert

        Args:
            question_text: The question stem
            choices: Dict of {label: text} for A/B/C/D/E
            ir_dict: IR dictionary for context

        Returns:
            CEGISResult with MCQ-specific certificate
        """
        import time
        t_start = time.perf_counter()
        trace = []
        trace.append(f"MCQ verification | {len(choices)} options")

        verified_options = []
        disproved_options = []
        unknown_options = []

        # Step 1: Create candidates for each option
        for label, text in sorted(choices.items()):
            # Create a candidate for this option
            cand = Candidate(
                value=label,
                construction=[f"mcq_option_{label}"],
                confidence=0.5,
                constraints=["option_label"],
                metadata={"option_text": text, "option_label": label},
            )

            # Step 2: Try to verify this option using world generation
            world_spec = self._infer_world_spec(ir_dict)
            worlds = self.world_gen.generate(world_spec.domain, world_spec.params)

            counterex = self._find_counterexample(cand, worlds, ir_dict)

            if counterex is None:
                # No counterexample found → this option might be correct
                if cand.metadata.get("verifier_pass"):
                    verified_options.append(label)
                    trace.append(f"  ✓ Option {label}: VERIFIED (verifier pass)")
                else:
                    unknown_options.append(label)
                    trace.append(f"  ? Option {label}: UNKNOWN (no counterexample, no verification)")
            else:
                # Counterexample found → this option is wrong
                disproved_options.append(label)
                trace.append(f"  ✗ Option {label}: DISPROVED ({counterex})")

        elapsed = (time.perf_counter() - t_start) * 1000

        # Step 3: Determine result based on verification status
        if len(verified_options) == 1:
            # Exactly one option verified → high confidence answer
            answer = verified_options[0]
            cert = Certificate(
                kind=CertKind.MCQ_VERIFIED,
                value={
                    "verified_option": answer,
                    "disproved_options": disproved_options,
                    "method": "option_by_option_verification",
                },
                confidence=0.90,
                details={"total_options": len(choices), "trace": trace},
            )
            return CEGISResult(
                answer=answer,
                confidence=0.90,
                certificate=cert,
                iterations=1,
                status="proved",
                trace=trace,
                elapsed_ms=elapsed,
            )

        elif len(disproved_options) == len(choices) - 1:
            # All but one option disproved → elimination proof
            remaining = [label for label in choices if label not in disproved_options][0]
            cert = Certificate(
                kind=CertKind.ELIMINATION,
                value={
                    "remaining_option": remaining,
                    "eliminated": disproved_options,
                    "total_options": len(choices),
                },
                confidence=0.85,
                details={"trace": trace},
            )
            return CEGISResult(
                answer=remaining,
                confidence=0.85,
                certificate=cert,
                iterations=1,
                status="proved",
                trace=trace,
                elapsed_ms=elapsed,
            )

        elif len(verified_options) > 1:
            # Multiple options verified → ambiguous, return highest confidence unknown
            trace.append(f"AMBIGUOUS: {len(verified_options)} options verified")
            return CEGISResult(
                answer=None,
                confidence=0.0,
                certificate=None,
                iterations=1,
                status="unknown",
                trace=trace,
                elapsed_ms=elapsed,
            )

        else:
            # No clear winner → return unknown
            trace.append(f"INCONCLUSIVE: {len(unknown_options)} unknown, {len(disproved_options)} disproved")
            return CEGISResult(
                answer=None,
                confidence=0.0,
                certificate=None,
                iterations=1,
                status="unknown",
                trace=trace,
                elapsed_ms=elapsed,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ファクトリ関数
# ─────────────────────────────────────────────────────────────────────────────


def make_candidates_from_executor_result(
    result: Any,
    piece_id: str,
    confidence: float = 0.8,
    constraints: Optional[List[str]] = None,
) -> List[Candidate]:
    """
    Executor の実行結果から Candidate リストを生成するヘルパー

    既存の Executor システムとの接続点。
    """
    if result is None:
        return []
    if isinstance(result, list):
        return [
            Candidate(
                value=v,
                construction=[piece_id],
                confidence=confidence * (0.9 ** i),
                constraints=constraints or [],
            )
            for i, v in enumerate(result[:5])
        ]
    return [
        Candidate(
            value=result,
            construction=[piece_id],
            confidence=confidence,
            constraints=constraints or [],
        )
    ]
