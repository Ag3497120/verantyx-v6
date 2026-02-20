"""
Algebra Executor - 代数演算

方程式を解く、多項式を評価する、など
"""
from typing import Optional, Union, List, Dict, Any
import re


def solve_equation(equation: str) -> Optional[float]:
    """
    方程式を解く（algebra_solve_equationのエイリアス）
    
    線形方程式 ax + b = c を解く
    
    Args:
        equation: 方程式文字列（例: "2x + 3 = 7"）
    
    Returns:
        解（xの値）、解けない場合None
    """
    # equation_solver.pyのsolve_linear_equationを使用
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from executors.equation_solver import solve_linear_equation
    return solve_linear_equation(equation)


def algebra_solve_equation(equation: str) -> Optional[float]:
    """
    代数方程式を解く（HLEで呼ばれる名前）
    
    Args:
        equation: 方程式文字列
    
    Returns:
        解、解けない場合None
    """
    return solve_equation(equation)


def evaluate_polynomial(expression: str = '', x: float = 0.0) -> Optional[float]:
    """
    多項式を評価する
    
    Args:
        expression: 多項式文字列（例: "x^2 + 2*x + 1"）
        x: 変数xの値
    
    Returns:
        評価結果
    """
    try:
        # xを値に置換
        expr = expression.replace('x', str(x))
        # ^をPythonの**に変換
        expr = re.sub(r'\^', '**', expr)
        # 暗黙的な乗算を明示的に（2xを2*xに）
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        # 評価
        result = eval(expr)
        return float(result)
    except Exception as e:
        return None


def simplify_expression(expression: str) -> Optional[str]:
    """
    式を簡約する（基本的な変形のみ）
    
    Args:
        expression: 代数式
    
    Returns:
        簡約された式、または入力が長すぎる/式でない場合None
    """
    if not expression:
        return None
    
    # 長すぎる入力は問題文全体が渡されている可能性 → None を返す
    if len(expression) > 150:
        return None
    
    # 数学記号がほぼない場合は式ではない
    if not re.search(r'[+\-*/=^x0-9]', expression):
        return None
    
    # 基本的な整理のみ
    expr = expression.strip()
    expr = expr.replace(' ', '')
    expr = re.sub(r'\+\+', '+', expr)
    expr = re.sub(r'--', '+', expr)
    expr = re.sub(r'\+-', '-', expr)
    
    return expr


def solve_linear_equation(equation: str = None, variable: str = 'x') -> Optional[float]:
    """
    一次方程式を解く（algebra_solve_linearピース用）
    例: "2x + 4 = 0" → -2.0

    Args:
        equation: 方程式の文字列（例: "2*x + 4 = 0", "3x = 9"）
        variable: 変数名（デフォルト: 'x'）

    Returns:
        解（float）、または解けない場合 None
    """
    try:
        from sympy import symbols, solve as sp_solve, sympify
        import re as _re

        if not equation or len(equation) > 300:
            return None

        eq = equation.strip()
        var = variable or 'x'
        sym = symbols(var)

        # 暗黙的な乗算を明示化: 2x → 2*x
        eq = _re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq)
        # 括弧前の係数も: 2(x+1) → 2*(x+1)
        eq = _re.sub(r'(\d)(\()', r'\1*\2', eq)

        # 等号で分割して左辺 - 右辺 の形式にする
        if '=' in eq:
            lhs, rhs = eq.split('=', 1)
            expr = sympify(lhs.strip()) - sympify(rhs.strip())
        else:
            expr = sympify(eq)

        result = sp_solve(expr, sym)
        if result:
            val = complex(result[0])
            if abs(val.imag) < 1e-9:
                return float(val.real)
        return None
    except Exception:
        return None


def factor_polynomial(expression: str = None) -> Optional[str]:
    """
    多項式を因数分解する（algebra_factorピース用）

    Args:
        expression: 多項式（例: "x^2 - 5*x + 6"）

    Returns:
        因数分解された式の文字列、または失敗時 None
    """
    try:
        from sympy import factor, sympify

        if not expression or len(expression) > 300:
            return None

        expr = sympify(expression.replace('^', '**'))
        factored = factor(expr)
        return str(factored)
    except Exception:
        return None


def partition_number(n: int = None) -> Optional[int]:
    """
    整数の分割数 p(n) を計算（partition_numピース用）
    """
    try:
        if n is None:
            return None
        n = int(n)
        if n < 0 or n > 200:
            return None
        # 動的プログラミングで計算
        dp = [0] * (n + 1)
        dp[0] = 1
        for k in range(1, n + 1):
            for i in range(k, n + 1):
                dp[i] += dp[i - k]
        return dp[n]
    except Exception:
        return None


def main():
    """テスト実行"""
    print("=" * 80)
    print("Algebra Executor Test")
    print("=" * 80)
    print()
    
    # solve_equation テスト
    print("[Equation Solving]")
    eq_tests = [
        ("2x + 3 = 7", 2.0),
        ("x - 5 = 10", 15.0),
        ("3x = 12", 4.0)
    ]
    
    for eq, expected in eq_tests:
        result = algebra_solve_equation(eq)
        status = "✅" if result is not None and abs(result - expected) < 1e-6 else "❌"
        print(f"{status} algebra_solve_equation('{eq}') = {result} (expected: {expected})")
    
    print()
    
    # evaluate_polynomial テスト
    print("[Polynomial Evaluation]")
    poly_tests = [
        ("x^2 + 2*x + 1", 3, 16.0),  # (3)^2 + 2*3 + 1 = 16
        ("2*x", 5, 10.0),
        ("x", 7, 7.0)
    ]
    
    for expr, x, expected in poly_tests:
        result = evaluate_polynomial(expr, x)
        status = "✅" if result is not None and abs(result - expected) < 1e-6 else "❌"
        print(f"{status} evaluate_polynomial('{expr}', {x}) = {result} (expected: {expected})")
    
    print()
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
