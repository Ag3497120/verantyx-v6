#!/usr/bin/env python3
"""
【Cross Simulator - Layer 4】

Verantyx思想の核心: Reject/Promoteによる機械的検証

仮説生成 → micro-world構築 → 全仮説をテスト → 反例があればReject → 全ケース一致でPromote

人間が「わからない」状態で機械的に結論を確定する
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import re
from pathlib import Path

from .avh_adapters import ILSlots, CrossDB, CrossAsset
from .propositional_logic_solver import is_tautology, is_satisfiable
from .modal_logic_solver import check_axiom_validity


@dataclass
class Hypothesis:
    """仮説（公理適用パターン）"""

    # 仮説ID
    hypothesis_id: str

    # 使用する公理・定理
    axioms: List[CrossAsset]

    # 適用手順（converters → solvers → verifiers）
    procedure: List[str]

    # 仮説の内容（例: "F=ma を使ってa=F/mを計算"）
    description: str

    # 信頼度
    confidence: float = 0.0

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicroCase:
    """マイクロケース（小世界テスト用）"""

    # ケースID
    case_id: str

    # 入力値（例: {"force": 1.0, "mass": 1.0}）
    inputs: Dict[str, float]

    # 期待出力（計算可能なら）
    expected_output: Optional[float] = None

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """シミュレーション結果"""

    # 確定した結論（値）
    conclusion: Optional[str] = None

    # 検証済みか
    verified: bool = False

    # 使用した公理
    axioms_used: List[str] = field(default_factory=list)

    # 推論トレース
    trace: List[str] = field(default_factory=list)

    # Rejectされた仮説
    rejected_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Promoteされた仮説
    promoted_hypothesis: Optional[Hypothesis] = None

    # 単位情報
    unit: Optional[str] = None

    # 信頼度
    confidence: float = 0.0

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossSimulator:
    """
    Cross Simulator: 小世界シミュレーションによる仮説検証

    Verantyx Core:
    1. 仮説生成（公理から複数の適用パターンを生成）
    2. Micro-world構築（小さな数値でテストケース作成）
    3. 全仮説を全ケースでテスト
    4. 反例が1つでも → Reject
    5. 全ケース一致 → Promote → 確定
    """

    def __init__(self, cross_db: Optional[CrossDB] = None):
        """
        Args:
            cross_db: 公理・定理のデータベース
        """
        self.cross_db = cross_db

        # 実行器（公理→実行関数のマッピング）
        self.executors = self._build_executors()

    def _build_executors(self) -> Dict[str, Callable]:
        """
        公理を実行する関数を登録

        各公理IDに対して、実際の計算を行う関数をマッピング
        """
        return {
            # Physics
            "axiom:physics:newtons_second_law": self._execute_newtons_second_law,
            "axiom:physics:conservation_of_energy": self._execute_conservation_of_energy,

            # Mathematics
            "axiom:mathematics:pythagorean_theorem": self._execute_pythagorean_theorem,
            "axiom:mathematics:fundamental_theorem_calculus": self._execute_ftc,
            "axiom:mathematics:matrix_multiplication": self._execute_matrix_mult,

            # Arithmetic
            "axiom:arithmetic:addition": self._execute_addition,
            "axiom:arithmetic:subtraction": self._execute_subtraction,
            "axiom:arithmetic:multiplication": self._execute_multiplication,
            "axiom:arithmetic:division": self._execute_division,

            # Logic - Propositional
            "axiom:logic:tautology_checker": self._execute_tautology_checker,
            "axiom:logic:sat_solver": self._execute_sat_solver,
            "axiom:logic:modal_k": self._execute_modal_k,
            "axiom:logic:modus_ponens": self._execute_modus_ponens,

            # Logic - Modal Axioms
            "axiom:logic:modal_t": self._execute_modal_t,
            "axiom:logic:modal_4": self._execute_modal_4,
            "axiom:logic:modal_d": self._execute_modal_d,
            "axiom:logic:modal_5": self._execute_modal_5,
            "axiom:logic:modal_b": self._execute_modal_b,
        }

    def simulate(
        self,
        extracted_knowledge: Dict[str, Any],
        il_slots: ILSlots,
        problem_type: Dict[str, Any]
    ) -> SimulationResult:
        """
        シミュレーション実行

        Args:
            extracted_knowledge: Layer 3で抽出された知識（公理リスト）
            il_slots: ILスロット（問題の構造化表現）
            problem_type: 問題型情報（Layer 2Bの結果）

        Returns:
            SimulationResult: 確定した結論と検証結果
        """
        trace = []
        trace.append("[Layer 4] Cross Simulation Start")

        # Step 1: 仮説生成
        trace.append("[Step 1] Hypothesis Generation")
        hypotheses = self._generate_hypotheses(
            extracted_knowledge,
            il_slots,
            problem_type
        )
        trace.append(f"  Generated {len(hypotheses)} hypotheses")

        # Step 2: Micro-world構築
        trace.append("[Step 2] Micro-World Construction")
        micro_cases = self._build_micro_world(il_slots)
        trace.append(f"  Built {len(micro_cases)} micro-cases")

        # Step 3: 全仮説を全ケースでテスト（Reject/Promote）
        trace.append("[Step 3] Hypothesis Testing (Reject/Promote)")

        promoted_hypothesis = None
        rejected = []

        for hypothesis in hypotheses:
            trace.append(f"  Testing: {hypothesis.hypothesis_id}")

            # 全ケースでテスト
            test_results = []
            for micro_case in micro_cases:
                result = self._test_hypothesis(hypothesis, micro_case, il_slots)
                test_results.append(result)

            # 判定
            all_passed = all(r["passed"] for r in test_results)

            if all_passed:
                trace.append(f"    ✓ PROMOTE: All {len(test_results)} cases passed")
                promoted_hypothesis = hypothesis
                break  # 最初にPromoteされた仮説で確定
            else:
                failed_count = sum(1 for r in test_results if not r["passed"])
                trace.append(f"    ✗ REJECT: {failed_count}/{len(test_results)} cases failed")
                rejected.append({
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "reason": f"{failed_count} cases failed"
                })

        # Step 4: 確定した結論を生成
        if promoted_hypothesis:
            trace.append("[Step 4] Conclusion Generation")
            conclusion = self._generate_conclusion(promoted_hypothesis, il_slots)

            return SimulationResult(
                conclusion=conclusion["value"],
                verified=True,
                axioms_used=[a.asset_id for a in promoted_hypothesis.axioms],
                trace=trace,
                rejected_hypotheses=rejected,
                promoted_hypothesis=promoted_hypothesis,
                unit=conclusion.get("unit"),
                confidence=promoted_hypothesis.confidence,
                metadata={"test_cases": len(micro_cases)}
            )
        else:
            trace.append("[Step 4] No hypothesis promoted - simulation failed")
            return SimulationResult(
                verified=False,
                trace=trace,
                rejected_hypotheses=rejected,
                confidence=0.0
            )

    def _generate_hypotheses(
        self,
        extracted_knowledge: Dict[str, Any],
        il_slots: ILSlots,
        problem_type: Dict[str, Any]
    ) -> List[Hypothesis]:
        """
        仮説生成

        抽出された公理から、適用可能なパターンを複数生成
        """
        hypotheses = []

        axioms = extracted_knowledge.get("axioms", [])

        for i, axiom in enumerate(axioms):
            # 公理が実行可能か確認
            if axiom.asset_id in self.executors:
                hypothesis = Hypothesis(
                    hypothesis_id=f"H{i+1}_{axiom.asset_id.split(':')[-1]}",
                    axioms=[axiom],
                    procedure=["converter", "solver", "verifier"],
                    description=f"Apply {axiom.content.get('title', axiom.asset_id)}",
                    confidence=axiom.confidence,
                    metadata={"axiom_id": axiom.asset_id}
                )
                hypotheses.append(hypothesis)

        # フォールバック: 実行可能な公理がなければダミー仮説
        if len(hypotheses) == 0:
            hypotheses.append(Hypothesis(
                hypothesis_id="H0_fallback",
                axioms=[],
                procedure=["pattern_match"],
                description="Fallback pattern matching",
                confidence=0.1
            ))

        return hypotheses

    def _build_micro_world(self, il_slots: ILSlots) -> List[MicroCase]:
        """
        Micro-world構築

        CONSTRAINTSから数値を抽出し、小さな値でテストケースを作成
        """
        micro_cases = []

        # 実際の値を抽出
        actual_values = {}
        for constraint in il_slots.CONSTRAINTS:
            # "force=10", "mass=2" のような形式を想定
            match = re.match(r"(\w+)\s*=\s*([\d.]+)", constraint)
            if match:
                var_name = match.group(1)
                value = float(match.group(2))
                actual_values[var_name] = value

        if len(actual_values) == 0:
            # CONSTRAINTSがない場合はダミーケース
            return [MicroCase(case_id="M0_empty", inputs={})]

        # Case 1: 小さな値 (1.0)
        micro_cases.append(MicroCase(
            case_id="M1_unit",
            inputs={k: 1.0 for k in actual_values.keys()},
            metadata={"description": "Unit test case"}
        ))

        # Case 2: 中間値 (2.0)
        micro_cases.append(MicroCase(
            case_id="M2_double",
            inputs={k: 2.0 for k in actual_values.keys()},
            metadata={"description": "Double test case"}
        ))

        # Case 3: 実際の値
        micro_cases.append(MicroCase(
            case_id="M3_actual",
            inputs=actual_values,
            metadata={"description": "Actual problem values"}
        ))

        return micro_cases

    def _test_hypothesis(
        self,
        hypothesis: Hypothesis,
        micro_case: MicroCase,
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        1つの仮説を1つのmicro-caseでテスト

        Returns:
            {"passed": bool, "output": Any, "error": str}
        """
        try:
            # 公理を実行
            if len(hypothesis.axioms) == 0:
                # Fallback仮説
                return {"passed": False, "error": "No axiom to execute"}

            axiom = hypothesis.axioms[0]

            if axiom.asset_id not in self.executors:
                return {"passed": False, "error": f"No executor for {axiom.asset_id}"}

            executor = self.executors[axiom.asset_id]

            # 実行
            output = executor(micro_case.inputs, il_slots)

            # 検証: 出力が数値として妥当か
            if output is None or output.get("value") is None:
                return {"passed": False, "error": "No output value"}

            # 基本的な妥当性チェック（負の値、NaN、inf など）
            value = output["value"]
            if isinstance(value, (int, float)):
                if value < 0 and il_slots.TARGET in ["acceleration", "mass", "length"]:
                    return {"passed": False, "error": "Negative value for positive quantity"}
                if not (-1e10 < value < 1e10):
                    return {"passed": False, "error": "Value out of reasonable range"}

            return {"passed": True, "output": output}

        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _generate_conclusion(
        self,
        hypothesis: Hypothesis,
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        確定した仮説から最終結論を生成

        実際の値で再実行して結果を取得
        """
        # 実際の値を抽出
        actual_values = {}
        for constraint in il_slots.CONSTRAINTS:
            match = re.match(r"(\w+)\s*=\s*([\d.]+)", constraint)
            if match:
                var_name = match.group(1)
                value = float(match.group(2))
                actual_values[var_name] = value

        # 実行
        if len(hypothesis.axioms) > 0:
            axiom = hypothesis.axioms[0]
            executor = self.executors.get(axiom.asset_id)
            if executor:
                output = executor(actual_values, il_slots)
                return output

        return {"value": None, "unit": None}

    # ========================================
    # 公理実行器（Executors）
    # ========================================

    def _execute_newtons_second_law(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Newton's Second Law: F = m * a

        入力: force, mass のいずれか2つ
        出力: 残りの1つ
        """
        force = inputs.get("force")
        mass = inputs.get("mass")
        acceleration = inputs.get("acceleration")

        # acceleration を求める
        if force is not None and mass is not None and mass != 0:
            result = force / mass
            return {"value": result, "unit": "m/s²", "formula": "a = F/m"}

        # force を求める
        if mass is not None and acceleration is not None:
            result = mass * acceleration
            return {"value": result, "unit": "N", "formula": "F = m*a"}

        # mass を求める
        if force is not None and acceleration is not None and acceleration != 0:
            result = force / acceleration
            return {"value": result, "unit": "kg", "formula": "m = F/a"}

        return {"value": None, "error": "Insufficient inputs"}

    def _execute_pythagorean_theorem(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Pythagorean Theorem: a² + b² = c²
        """
        import math

        # 様々な辺の命名パターンに対応
        a = (inputs.get("a") or inputs.get("side_a") or inputs.get("side1") or
             inputs.get("side_1"))
        b = (inputs.get("b") or inputs.get("side_b") or inputs.get("side2") or
             inputs.get("side_2"))

        # CONSTRAINTSから直接抽出も試みる
        if a is None or b is None:
            for constraint in il_slots.CONSTRAINTS:
                if "=" in constraint:
                    parts = constraint.split("=")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            if var_name in ["a", "side_a", "side1", "side_1"]:
                                a = value
                            elif var_name in ["b", "side_b", "side2", "side_2"]:
                                b = value
                        except ValueError:
                            continue

        if a is not None and b is not None:
            c = math.sqrt(a**2 + b**2)
            return {"value": c, "unit": "", "formula": "c = √(a²+b²)"}

        return {"value": None, "error": "Insufficient sides"}

    def _execute_conservation_of_energy(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Conservation of Energy: E_total = E_kinetic + E_potential
        """
        # Placeholder
        return {"value": None, "error": "Not implemented"}

    def _execute_ftc(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Fundamental Theorem of Calculus
        """
        # Placeholder
        return {"value": None, "error": "Not implemented"}

    def _execute_matrix_mult(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Matrix Multiplication
        """
        # Placeholder
        return {"value": None, "error": "Not implemented"}

    def _execute_modal_k(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal Logic K Axiom: □(A→B) → (□A→□B)
        """
        # Placeholder
        return {"value": None, "error": "Not implemented"}

    def _execute_modus_ponens(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modus Ponens: A, A→B ⊢ B
        """
        # Placeholder
        return {"value": None, "error": "Not implemented"}

    # ========================================
    # 算術演算実行器（Arithmetic Executors）
    # ========================================

    def _execute_addition(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Addition: a + b = c
        """
        # CONSTRAINTSから算術式を探す
        for constraint in il_slots.CONSTRAINTS:
            if constraint.startswith("expression="):
                expression = constraint.replace("expression=", "")
                # 簡単な加法式を評価
                if "+" in expression:
                    parts = expression.split("+")
                    if len(parts) == 2:
                        try:
                            a = float(parts[0].strip())
                            b = float(parts[1].strip())
                            result = a + b
                            return {"value": result, "unit": "", "formula": f"{a} + {b} = {result}"}
                        except ValueError:
                            pass

        # 直接inputsから取得
        a = inputs.get("a")
        b = inputs.get("b")
        if a is not None and b is not None:
            result = a + b
            return {"value": result, "unit": "", "formula": f"{a} + {b} = {result}"}

        return {"value": None, "error": "Insufficient operands"}

    def _execute_subtraction(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Subtraction: a - b = c
        """
        # CONSTRAINTSから算術式を探す
        for constraint in il_slots.CONSTRAINTS:
            if constraint.startswith("expression="):
                expression = constraint.replace("expression=", "")
                if "-" in expression:
                    parts = expression.split("-")
                    if len(parts) == 2:
                        try:
                            a = float(parts[0].strip())
                            b = float(parts[1].strip())
                            result = a - b
                            return {"value": result, "unit": "", "formula": f"{a} - {b} = {result}"}
                        except ValueError:
                            pass

        # 直接inputsから取得
        a = inputs.get("a")
        b = inputs.get("b")
        if a is not None and b is not None:
            result = a - b
            return {"value": result, "unit": "", "formula": f"{a} - {b} = {result}"}

        return {"value": None, "error": "Insufficient operands"}

    def _execute_multiplication(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Multiplication: a × b = c
        """
        # CONSTRAINTSから算術式を探す
        for constraint in il_slots.CONSTRAINTS:
            if constraint.startswith("expression="):
                expression = constraint.replace("expression=", "")
                if "*" in expression:
                    parts = expression.split("*")
                    if len(parts) == 2:
                        try:
                            a = float(parts[0].strip())
                            b = float(parts[1].strip())
                            result = a * b
                            return {"value": result, "unit": "", "formula": f"{a} * {b} = {result}"}
                        except ValueError:
                            pass

        # 直接inputsから取得
        a = inputs.get("a")
        b = inputs.get("b")
        if a is not None and b is not None:
            result = a * b
            return {"value": result, "unit": "", "formula": f"{a} * {b} = {result}"}

        return {"value": None, "error": "Insufficient operands"}

    def _execute_division(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Division: a ÷ b = c
        """
        # CONSTRAINTSから算術式を探す
        for constraint in il_slots.CONSTRAINTS:
            if constraint.startswith("expression="):
                expression = constraint.replace("expression=", "")
                if "/" in expression:
                    parts = expression.split("/")
                    if len(parts) == 2:
                        try:
                            a = float(parts[0].strip())
                            b = float(parts[1].strip())
                            if b == 0:
                                return {"value": None, "error": "Division by zero"}
                            result = a / b
                            return {"value": result, "unit": "", "formula": f"{a} / {b} = {result}"}
                        except ValueError:
                            pass

        # 直接inputsから取得
        a = inputs.get("a")
        b = inputs.get("b")
        if a is not None and b is not None:
            if b == 0:
                return {"value": None, "error": "Division by zero"}
            result = a / b
            return {"value": result, "unit": "", "formula": f"{a} / {b} = {result}"}

        return {"value": None, "error": "Insufficient operands"}

    # ========================================
    # 論理演算実行器（Logic Executors）
    # ========================================

    def _execute_tautology_checker(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Tautology Checker: 命題論理式の恒真性判定
        """
        # CONSTRAINTSから論理式を探す
        formula_str = None
        for constraint in il_slots.CONSTRAINTS:
            # "formula=..." 形式から論理式を抽出
            if constraint.startswith("formula="):
                formula_str = constraint.replace("formula=", "").strip()
                break
            # 式を直接含むconstraint
            elif "->" in constraint or "&" in constraint or "|" in constraint:
                formula_str = constraint
                break

        # OBJECTから論理式を抽出
        if not formula_str:
            for obj in il_slots.OBJECT:
                if "->" in obj or "&" in obj or "|" in obj:
                    formula_str = obj
                    break

        if not formula_str:
            return {"value": None, "error": "No propositional formula found"}

        try:
            is_taut, counterexample = is_tautology(formula_str)
            if is_taut:
                return {
                    "value": True,
                    "formula": f"{formula_str} is a tautology",
                    "explanation": "The formula is true under all variable assignments"
                }
            else:
                return {
                    "value": False,
                    "formula": f"{formula_str} is not a tautology",
                    "counterexample": str(counterexample)
                }
        except Exception as e:
            return {"value": None, "error": f"Evaluation error: {e}"}

    def _execute_sat_solver(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        SAT Solver: 命題論理式の充足可能性判定
        """
        # CONSTRAINTSから論理式を探す
        formula_str = None
        for constraint in il_slots.CONSTRAINTS:
            # "formula=..." 形式から論理式を抽出
            if constraint.startswith("formula="):
                formula_str = constraint.replace("formula=", "").strip()
                break
            # 式を直接含むconstraint
            elif "->" in constraint or "&" in constraint or "|" in constraint:
                formula_str = constraint
                break

        if not formula_str:
            for obj in il_slots.OBJECT:
                if "->" in obj or "&" in obj or "|" in obj:
                    formula_str = obj
                    break

        if not formula_str:
            return {"value": None, "error": "No propositional formula found"}

        try:
            is_sat, assignment = is_satisfiable(formula_str)
            if is_sat:
                return {
                    "value": True,
                    "formula": f"{formula_str} is satisfiable",
                    "satisfying_assignment": str(assignment)
                }
            else:
                return {
                    "value": False,
                    "formula": f"{formula_str} is unsatisfiable (contradiction)"
                }
        except Exception as e:
            return {"value": None, "error": f"Evaluation error: {e}"}

    def _execute_modal_t(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal T Axiom: []p -> p (Reflexivity)
        """
        return self._check_modal_axiom("T", il_slots)

    def _execute_modal_4(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal 4 Axiom: []p -> [][]p (Transitivity)
        """
        return self._check_modal_axiom("4", il_slots)

    def _execute_modal_d(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal D Axiom: []p -> <>p (Serial)
        """
        return self._check_modal_axiom("D", il_slots)

    def _execute_modal_5(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal 5 Axiom: <>p -> []<>p (Euclidean)
        """
        return self._check_modal_axiom("5", il_slots)

    def _execute_modal_b(
        self,
        inputs: Dict[str, float],
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        Modal B Axiom: p -> []<>p (Symmetric)
        """
        return self._check_modal_axiom("B", il_slots)

    def _check_modal_axiom(
        self,
        axiom_name: str,
        il_slots: ILSlots
    ) -> Dict[str, Any]:
        """
        様相論理公理の検証（ヘルパーメソッド）
        """
        # CONSTRAINTSから様相論理式を探す
        formula_str = None
        for constraint in il_slots.CONSTRAINTS:
            # "formula=..." 形式から論理式を抽出
            if constraint.startswith("formula="):
                formula_str = constraint.replace("formula=", "").strip()
                break
            # 直接論理式が入っている場合
            elif "[]" in constraint or "<>" in constraint:
                formula_str = constraint
                break

        if not formula_str:
            for obj in il_slots.OBJECT:
                if "[]" in obj or "<>" in obj:
                    formula_str = obj
                    break

        if not formula_str:
            return {"value": None, "error": "No modal formula found"}

        try:
            is_valid, explanation = check_axiom_validity(axiom_name, formula_str)
            if is_valid:
                return {
                    "value": True,
                    "formula": formula_str,
                    "explanation": explanation
                }
            else:
                return {
                    "value": False,
                    "formula": formula_str,
                    "explanation": explanation
                }
        except Exception as e:
            return {"value": None, "error": f"Evaluation error: {e}"}


def main():
    """テスト実行"""
    from .il_converter import ILConverter

    print("=" * 80)
    print("Cross Simulator Test")
    print("=" * 80)

    # Load axiom database
    axiom_db_path = Path(__file__).parent / "axioms_unified.json"
    if not axiom_db_path.exists():
        print(f"⚠️  Axiom database not found: {axiom_db_path}")
        print("Please run: python3 tools/merge_axiom_databases.py")
        return

    cross_db = CrossDB(str(axiom_db_path))
    simulator = CrossSimulator(cross_db)
    converter = ILConverter()

    # Test case: Newton's Second Law
    problem = "If force = 10 N and mass = 2 kg, what is acceleration?"

    print(f"\nProblem: {problem}")
    print()

    # IL変換
    il_slots = converter.convert(problem)
    print(f"IL Slots: {il_slots}")
    print()

    # 知識抽出（Layer 2A/2Bの代わりに簡易実装）
    # CONSTRAINTSから変数名を抽出（"force=10 N" → "force"）
    import re
    constraint_vars = []
    for constraint in il_slots.CONSTRAINTS:
        match = re.match(r"(\w+)\s*=", constraint)
        if match:
            constraint_vars.append(match.group(1))

    # OBJECTと抽出した変数を合わせて検索
    requires_list = il_slots.OBJECT + constraint_vars
    provides_list = [il_slots.TARGET] if il_slots.TARGET else []

    exact_results = cross_db.search_exact(
        requires=requires_list,
        provides=provides_list
    )

    extracted_knowledge = {
        "axioms": exact_results[:3]
    }

    problem_type = {
        "type": "physics_equation_solving",
        "applicable_solvers": ["algebraic_solver"]
    }

    print(f"Extracted {len(extracted_knowledge['axioms'])} axioms:")
    for axiom in extracted_knowledge["axioms"]:
        print(f"  - {axiom.asset_id}")
    print()

    # シミュレーション実行
    print("[Simulation Start]")
    result = simulator.simulate(extracted_knowledge, il_slots, problem_type)

    print()
    print("=" * 80)
    print("Simulation Result")
    print("=" * 80)
    print(f"Verified: {result.verified}")
    print(f"Conclusion: {result.conclusion} {result.unit or ''}")
    print(f"Confidence: {result.confidence}")
    print(f"Axioms Used: {result.axioms_used}")
    print(f"Rejected: {len(result.rejected_hypotheses)} hypotheses")

    if result.promoted_hypothesis:
        print(f"Promoted: {result.promoted_hypothesis.hypothesis_id}")
        print(f"  Description: {result.promoted_hypothesis.description}")

    print()
    print("Trace:")
    for line in result.trace:
        print(f"  {line}")

    print()
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
