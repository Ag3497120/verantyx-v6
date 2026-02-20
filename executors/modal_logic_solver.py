#!/usr/bin/env python3
"""
【Modal Logic Solver】

様相論理の検証器・解法器

実装:
- Kripke Frame (フレーム構造)
- Kripke Model (モデル構造)
- Modal Formula Evaluation (様相論理式の評価)
- Frame Property Checkers (フレーム特性の検証)
- Validity Checker (妥当性判定)
"""
from typing import Dict, Set, List, Optional, Tuple
import re
from itertools import product


class KripkeFrame:
    """Kripkeフレーム"""

    def __init__(self, worlds: Set[str], relation: Set[Tuple[str, str]]):
        """
        Args:
            worlds: 可能世界の集合
            relation: アクセス関係の集合 (w1, w2) = w1からw2へアクセス可能
        """
        self.worlds = worlds
        self.relation = relation

    def accessible(self, w: str) -> Set[str]:
        """世界wからアクセス可能な世界の集合"""
        return {w2 for w1, w2 in self.relation if w1 == w}

    def is_reflexive(self) -> bool:
        """反射性: すべてのwについて (w, w) ∈ R"""
        return all((w, w) in self.relation for w in self.worlds)

    def is_transitive(self) -> bool:
        """推移性: (w1, w2) ∈ R かつ (w2, w3) ∈ R ならば (w1, w3) ∈ R"""
        for w1, w2 in self.relation:
            for w2_, w3 in self.relation:
                if w2 == w2_ and (w1, w3) not in self.relation:
                    return False
        return True

    def is_serial(self) -> bool:
        """系列性: すべてのwについて、少なくとも1つのw'が存在して (w, w') ∈ R"""
        return all(len(self.accessible(w)) > 0 for w in self.worlds)

    def is_euclidean(self) -> bool:
        """ユークリッド性: (w, w1) ∈ R かつ (w, w2) ∈ R ならば (w1, w2) ∈ R"""
        for w in self.worlds:
            accessible = self.accessible(w)
            for w1 in accessible:
                for w2 in accessible:
                    if (w1, w2) not in self.relation:
                        return False
        return True

    def is_symmetric(self) -> bool:
        """対称性: (w1, w2) ∈ R ならば (w2, w1) ∈ R"""
        return all((w2, w1) in self.relation for w1, w2 in self.relation)


class ModalFormula:
    """様相論理式の簡易表現"""

    def __init__(self, formula_str: str):
        self.original = formula_str
        self.normalized = self._normalize(formula_str)
        self.variables = self._extract_variables(self.normalized)

    def _normalize(self, formula: str) -> str:
        """式を正規化"""
        # シンボルを統一
        formula = formula.replace("∧", "&")
        formula = formula.replace("∨", "|")
        formula = formula.replace("¬", "~")
        formula = formula.replace("→", "->")
        formula = formula.replace("□", "[]")
        formula = formula.replace("◇", "<>")
        formula = formula.replace("<>", "<>")  # possibility
        formula = formula.replace("[]", "[]")  # necessity
        return formula

    def _extract_variables(self, formula: str) -> Set[str]:
        """変数を抽出（命題変数のみ、演算子は除外）"""
        # Remove modal operators and other symbols first
        temp = formula.replace("[]", "").replace("<>", "")
        temp = temp.replace("&", " ").replace("|", " ").replace("~", " ")
        temp = temp.replace("(", " ").replace(")", " ").replace("->", " ")
        # Extract propositional variables
        variables = set(re.findall(r'\b[a-z]\d?\b', temp))
        return variables


class KripkeModel:
    """Kripkeモデル"""

    def __init__(self, frame: KripkeFrame, valuation: Dict[str, Set[str]]):
        """
        Args:
            frame: Kripkeフレーム
            valuation: 命題変数の真偽評価 {変数: その変数が真である世界の集合}
        """
        self.frame = frame
        self.valuation = valuation

    def evaluate(self, formula: ModalFormula, world: str) -> bool:
        """世界wにおいて式を評価"""
        return self._eval_at_world(formula.normalized, world)

    def _eval_at_world(self, expr: str, world: str) -> bool:
        """世界wにおいて式を再帰的に評価"""
        expr = expr.strip()

        # 命題変数
        if re.match(r'^[a-z]\d?$', expr):
            return world in self.valuation.get(expr, set())

        # 最も外側の括弧を除去
        if expr.startswith("(") and expr.endswith(")"):
            depth = 0
            for i, c in enumerate(expr):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break
            if i == len(expr) - 1:
                return self._eval_at_world(expr[1:-1], world)

        # [] (necessity)
        if expr.startswith("[]"):
            inner = expr[2:].strip()
            # すべてのアクセス可能な世界で真
            accessible = self.frame.accessible(world)
            if len(accessible) == 0:
                # アクセス可能な世界がない場合は真（空虚な真）
                return True
            return all(self._eval_at_world(inner, w) for w in accessible)

        # <> (possibility)
        if expr.startswith("<>"):
            inner = expr[2:].strip()
            # 少なくとも1つのアクセス可能な世界で真
            accessible = self.frame.accessible(world)
            if len(accessible) == 0:
                return False
            return any(self._eval_at_world(inner, w) for w in accessible)

        # 優先順位: -> | & ~

        # -> (implication)
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and i >= 1 and expr[i-1:i+1] == "->":
                left = expr[:i-1].strip()
                right = expr[i+1:].strip()
                return (not self._eval_at_world(left, world)) or self._eval_at_world(right, world)

        # | (or)
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and expr[i] == "|":
                left = expr[:i].strip()
                right = expr[i+1:].strip()
                return self._eval_at_world(left, world) or self._eval_at_world(right, world)

        # & (and)
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and expr[i] == "&":
                left = expr[:i].strip()
                right = expr[i+1:].strip()
                return self._eval_at_world(left, world) and self._eval_at_world(right, world)

        # ~ (not)
        if expr.startswith("~"):
            return not self._eval_at_world(expr[1:].strip(), world)

        raise ValueError(f"Cannot evaluate: {expr}")

    def is_valid(self, formula: ModalFormula) -> Tuple[bool, Optional[str]]:
        """
        モデルにおいて式が妥当か判定（全世界で真か）

        Returns:
            (is_valid: bool, counterexample_world: Optional[str])
        """
        for world in self.frame.worlds:
            if not self.evaluate(formula, world):
                return (False, world)
        return (True, None)


def check_axiom_validity(axiom_name: str, formula_str: str) -> Tuple[bool, str]:
    """
    既知の様相論理公理が妥当かチェック（簡易版）

    Args:
        axiom_name: 公理名 ("T", "4", "D", "5", "B", "K")
        formula_str: 様相論理式

    Returns:
        (is_valid: bool, explanation: str)
    """
    formula = ModalFormula(formula_str)

    # 公理ごとに必要なフレーム特性を確認
    axiom_checks = {
        "T": _check_t_axiom,
        "4": _check_4_axiom,
        "D": _check_d_axiom,
        "5": _check_5_axiom,
        "B": _check_b_axiom,
        "K": _check_k_axiom,
    }

    if axiom_name not in axiom_checks:
        return (False, f"Unknown axiom: {axiom_name}")

    return axiom_checks[axiom_name](formula)


def _check_t_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """T Axiom: []p -> p (reflexive frames)"""
    # Pattern: []X -> X
    pattern = r'^\[\](.+)->(.+)$'
    match = re.match(pattern, formula.normalized)
    if match and match.group(1).strip() == match.group(2).strip():
        return (True, "Valid T axiom: []p -> p")
    return (False, "Not a T axiom pattern")


def _check_4_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """4 Axiom: []p -> [][]p (transitive frames)"""
    # Pattern: []X -> [][]X
    pattern = r'^\[\](.+)->\[\]\[\](.+)$'
    match = re.match(pattern, formula.normalized)
    if match and match.group(1).strip() == match.group(2).strip():
        return (True, "Valid 4 axiom: []p -> [][]p")
    return (False, "Not a 4 axiom pattern")


def _check_d_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """D Axiom: []p -> <>p (serial frames)"""
    # Pattern: []X -> <>X
    pattern = r'^\[\](.+)-><>(.+)$'
    match = re.match(pattern, formula.normalized)
    if match and match.group(1).strip() == match.group(2).strip():
        return (True, "Valid D axiom: []p -> <>p")
    return (False, "Not a D axiom pattern")


def _check_5_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """5 Axiom: <>p -> []<>p (euclidean frames)"""
    # Pattern: <>X -> []<>X
    pattern = r'^<>(.+)->\[\]<>(.+)$'
    match = re.match(pattern, formula.normalized)
    if match and match.group(1).strip() == match.group(2).strip():
        return (True, "Valid 5 axiom: <>p -> []<>p")
    return (False, "Not a 5 axiom pattern")


def _check_b_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """B Axiom: p -> []<>p (symmetric frames)"""
    # Pattern: X -> []<>X
    pattern = r'^(.+)->\[\]<>(.+)$'
    match = re.match(pattern, formula.normalized)
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip()
        # Left should not have modal operators
        if "[]" not in left and "<>" not in left and left == right:
            return (True, "Valid B axiom: p -> []<>p")
    return (False, "Not a B axiom pattern")


def _check_k_axiom(formula: ModalFormula) -> Tuple[bool, str]:
    """K Axiom: [](p -> q) -> ([]p -> []q)"""
    # Pattern: [](X -> Y) -> ([]X -> []Y)
    # Simplified check: just verify it has the right structure
    if "[](" in formula.normalized and "->(" in formula.normalized:
        return (True, "Valid K axiom pattern (simplified check)")
    return (False, "Not a K axiom pattern")


def main():
    """テスト実行"""
    print("=" * 80)
    print("Modal Logic Solver Test")
    print("=" * 80)

    # Test axiom pattern recognition
    axiom_tests = [
        ("T", "[]p->p", True, "T Axiom: []p -> p"),
        ("4", "[]p->[][]p", True, "4 Axiom: []p -> [][]p"),
        ("D", "[]p-><>p", True, "D Axiom: []p -> <>p"),
        ("5", "<>p->[]<>p", True, "5 Axiom: <>p -> []<>p"),
        ("B", "p->[]<>p", True, "B Axiom: p -> []<>p"),
        ("K", "[](p->q)->([]p->[]q)", True, "K Axiom"),
        ("T", "[]p->[]p", False, "Not T axiom"),
    ]

    print("\n[Axiom Pattern Recognition Tests]")
    for axiom_name, formula, expected_valid, description in axiom_tests:
        print(f"\n[Test] {description}")
        print(f"  Formula: {formula}")
        is_valid, explanation = check_axiom_validity(axiom_name, formula)
        print(f"  Result: {explanation}")

        if is_valid == expected_valid:
            print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ INCORRECT (expected {'valid' if expected_valid else 'invalid'})")

    # Test Kripke model evaluation
    print("\n" + "=" * 80)
    print("\n[Kripke Model Evaluation Tests]")

    # Simple reflexive frame: w0 -> w0
    frame = KripkeFrame(
        worlds={"w0"},
        relation={("w0", "w0")}
    )

    # p is true at w0
    valuation = {"p": {"w0"}}
    model = KripkeModel(frame, valuation)

    test_formulas = [
        ("p", True, "p is true at w0"),
        ("[]p", True, "[]p is true (reflexive, p true everywhere)"),
        ("[]p->p", True, "T axiom holds"),
    ]

    for formula_str, expected, description in test_formulas:
        formula = ModalFormula(formula_str)
        result = model.evaluate(formula, "w0")
        print(f"\n[Test] {description}")
        print(f"  Formula: {formula_str}")
        print(f"  Result: {result}")

        if result == expected:
            print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ INCORRECT (expected {expected})")

    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
