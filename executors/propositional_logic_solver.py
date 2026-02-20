#!/usr/bin/env python3
"""
【Propositional Logic Solver】

命題論理の検証器・解法器

実装:
- Tautology Checker (真理値表による恒真性判定)
- SAT Solver (充足可能性判定)
- 簡易的な命題論理パーサー
"""
from typing import Dict, Set, List, Optional
import re
from itertools import product


class PropositionalFormula:
    """命題論理式の簡易表現"""

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
        formula = formula.replace("↔", "<->")
        formula = formula.replace("<>", "<->")  # 双条件
        return formula

    def _extract_variables(self, formula: str) -> Set[str]:
        """変数を抽出"""
        # A, B, C, A1, B2 などの変数を抽出
        variables = set(re.findall(r'\b[A-Z]\d?\b', formula))
        return variables

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """変数割り当てのもとで式を評価"""
        expr = self.normalized

        # 変数を値に置換
        for var in sorted(self.variables, key=len, reverse=True):
            val = assignment.get(var, False)
            expr = re.sub(r'\b' + var + r'\b', str(val), expr)

        # 再帰的に評価
        try:
            result = self._eval_expr(expr)
            return result
        except Exception as e:
            # パースエラー
            return False

    def _eval_expr(self, expr: str) -> bool:
        """式を再帰的に評価（演算子の優先順位を考慮）"""
        expr = expr.strip()

        # Base case: True/False
        if expr == "True":
            return True
        if expr == "False":
            return False

        # 最も外側の括弧を除去
        if expr.startswith("(") and expr.endswith(")"):
            # 対応する括弧かチェック
            depth = 0
            for i, c in enumerate(expr):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    # 最初の'('に対応する')'が最後ではない
                    break
            if i == len(expr) - 1:
                # 外側の括弧を除去
                return self._eval_expr(expr[1:-1])

        # 優先順位: <-> (最低) -> | & ~ (最高)
        # 最も優先順位の低い演算子から処理

        # <-> (biconditional) を探す
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and i >= 2 and expr[i-2:i+1] == "<->":
                left = expr[:i-2].strip()
                right = expr[i+1:].strip()
                return self._eval_expr(left) == self._eval_expr(right)

        # -> (implication) を探す
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and i >= 1 and expr[i-1:i+1] == "->":
                left = expr[:i-1].strip()
                right = expr[i+1:].strip()
                # A -> B = ~A | B
                return (not self._eval_expr(left)) or self._eval_expr(right)

        # | (or) を探す
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and expr[i] == "|":
                left = expr[:i].strip()
                right = expr[i+1:].strip()
                return self._eval_expr(left) or self._eval_expr(right)

        # & (and) を探す
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == "(":
                depth -= 1
            elif expr[i] == ")":
                depth += 1
            elif depth == 0 and expr[i] == "&":
                left = expr[:i].strip()
                right = expr[i+1:].strip()
                return self._eval_expr(left) and self._eval_expr(right)

        # ~ (not) を探す
        if expr.startswith("~"):
            return not self._eval_expr(expr[1:].strip())

        # どの演算子もない場合はエラー
        raise ValueError(f"Cannot evaluate: {expr}")


def is_tautology(formula_str: str) -> tuple:
    """
    恒真性判定

    Returns:
        (is_tautology: bool, counterexample: Optional[Dict])
    """
    formula = PropositionalFormula(formula_str)

    if len(formula.variables) == 0:
        # 変数がない場合（定数）
        result = formula.evaluate({})
        return (result, None if result else {})

    # すべての変数割り当てを試す
    for assignment_tuple in product([False, True], repeat=len(formula.variables)):
        assignment = dict(zip(sorted(formula.variables), assignment_tuple))

        if not formula.evaluate(assignment):
            # 反例が見つかった
            return (False, assignment)

    # 全ての割り当てでTrue → 恒真
    return (True, None)


def is_satisfiable(formula_str: str) -> tuple:
    """
    充足可能性判定

    Returns:
        (is_satisfiable: bool, satisfying_assignment: Optional[Dict])
    """
    formula = PropositionalFormula(formula_str)

    if len(formula.variables) == 0:
        # 変数がない場合
        result = formula.evaluate({})
        return (result, {} if result else None)

    # すべての変数割り当てを試す
    for assignment_tuple in product([False, True], repeat=len(formula.variables)):
        assignment = dict(zip(sorted(formula.variables), assignment_tuple))

        if formula.evaluate(assignment):
            # 充足する割り当てが見つかった
            return (True, assignment)

    # 充足可能な割り当てが見つからない
    return (False, None)


def main():
    """テスト実行"""
    print("=" * 80)
    print("Propositional Logic Solver Test")
    print("=" * 80)

    # Test cases
    test_cases = [
        # Tautologies
        ("((A -> B) & A) -> B", True, "Modus Ponens"),
        ("A | ~A", True, "Law of Excluded Middle"),
        ("(A -> B) <-> (~B -> ~A)", True, "Contraposition"),

        # Non-tautologies
        ("A & B", False, "Conjunction"),
        ("A -> B", False, "Implication"),

        # From HLE
        ("((A1->B1)&(A2->B2)&A1)->B1", True, "HLE Problem 25"),
        ("(((A->B)&(B->C)&(C->D)&A)->D)", True, "HLE Problem 27 (simplified)"),
    ]

    for formula, expected, description in test_cases:
        print(f"\n[Test] {description}")
        print(f"  Formula: {formula}")

        is_taut, counterexample = is_tautology(formula)
        print(f"  Is Tautology: {is_taut}")

        if is_taut == expected:
            print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ INCORRECT (expected {expected})")
            if counterexample:
                print(f"  Counterexample: {counterexample}")

    # Satisfiability tests
    print("\n" + "=" * 80)
    print("Satisfiability Tests")
    print("=" * 80)

    sat_test_cases = [
        ("A & B & C", True, "Simple conjunction"),
        ("A & ~A", False, "Contradiction"),
        ("((A1|A2)&(~A1)&(~A2))", False, "HLE Problem 26 (simplified)"),
    ]

    for formula, expected, description in sat_test_cases:
        print(f"\n[Test] {description}")
        print(f"  Formula: {formula}")

        is_sat, assignment = is_satisfiable(formula)
        print(f"  Is Satisfiable: {is_sat}")

        if is_sat == expected:
            print(f"  ✅ CORRECT")
            if assignment:
                print(f"  Satisfying Assignment: {assignment}")
        else:
            print(f"  ❌ INCORRECT (expected {expected})")

    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
