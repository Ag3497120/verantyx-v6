"""
sympy_solver.py
===============
SymPy による数式の直接求解

対象問題:
  - 代数方程式 (solve)
  - 微分 (diff)
  - 積分 (integrate)
  - 極限 (limit)
  - 式の簡略化 (simplify)
  - 数値評価 (evalf)

原則: SymPy は "計算器" として使う。出力は数値/式。
バイアスや暗記は使わない。
"""

import re
from typing import Any, Optional, List


def _parse_expr(expr_str: str):
    """
    文字列を SymPy 式にパースする。
    暗黙の乗算 (2x → 2*x) を補完して安全に評価。
    """
    import sympy
    from sympy.abc import x, y, z, n, k, t, a, b, c

    # 暗黙の乗算補完: 2x → 2*x, 3(x+1) → 3*(x+1)
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr_str)
    expr_str = expr_str.replace('^', '**')

    local_ns = {
        'x': x, 'y': y, 'z': z, 'n': n, 'k': k, 't': t,
        'a': a, 'b': b, 'c': c,
        'pi': sympy.pi, 'e': sympy.E,
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'exp': sympy.exp, 'log': sympy.log, 'sqrt': sympy.sqrt,
        'abs': sympy.Abs, 'Abs': sympy.Abs,
        'factorial': sympy.factorial,
    }
    return sympy.sympify(expr_str, locals=local_ns)


def _get_var(var_name: str = 'x'):
    """変数名から SymPy Symbol を取得"""
    import sympy
    return sympy.Symbol(var_name)


# ─────────────────────────────────────────────────────────────────────
# 代数方程式ソルバー
# ─────────────────────────────────────────────────────────────────────

def solve_equation(equation: str, variable: str = 'x') -> Optional[Any]:
    """
    代数方程式を解く。

    例:
      solve_equation("x^2 - 5*x + 6 = 0")  → [2, 3]
      solve_equation("2*x + 3 = 7")          → [2]
      solve_equation("x^2 - 4")             → [-2, 2]

    Returns:
      解のリスト、または単一解（1つの場合）、失敗時 None
    """
    import sympy
    try:
        var = _get_var(variable)

        # "=" を含む場合は lhs - rhs = 0 に変換
        if '=' in equation:
            parts = equation.split('=')
            lhs = _parse_expr(parts[0].strip())
            rhs = _parse_expr(parts[1].strip())
            expr = lhs - rhs
        else:
            expr = _parse_expr(equation)

        solutions = sympy.solve(expr, var)
        if not solutions:
            return None

        # 実数解のみ（複素数を除外）
        real_sols = [s for s in solutions if s.is_real]
        if not real_sols:
            real_sols = solutions  # 実数解なければ全解

        # 数値変換
        result = []
        for s in real_sols:
            try:
                fv = float(s.evalf())
                result.append(int(fv) if fv == int(fv) else round(fv, 6))
            except Exception:
                result.append(str(s))

        if len(result) == 1:
            return result[0]
        return sorted(result) if all(isinstance(r, (int, float)) for r in result) else result

    except Exception:
        return None


def solve_linear(a_coef: float, b_coef: float, c_val: float) -> Optional[float]:
    """
    線形方程式 ax + b = c を解く。

    例: solve_linear(2, 3, 7) → 2.0  (2x + 3 = 7)
    """
    try:
        if a_coef == 0:
            return None
        result = (c_val - b_coef) / a_coef
        return int(result) if result == int(result) else round(result, 6)
    except Exception:
        return None


def solve_quadratic(a_coef: float, b_coef: float, c_val: float) -> Optional[Any]:
    """
    二次方程式 ax^2 + bx + c = 0 を解く（判別式法）。

    例: solve_quadratic(1, -5, 6) → [2, 3]
    """
    import sympy
    try:
        x = sympy.Symbol('x')
        expr = a_coef * x**2 + b_coef * x + c_val
        solutions = sympy.solve(expr, x)
        real_sols = sorted([
            int(float(s)) if float(s) == int(float(s)) else round(float(s), 6)
            for s in solutions if s.is_real
        ])
        if len(real_sols) == 1:
            return real_sols[0]
        return real_sols if real_sols else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# 微分
# ─────────────────────────────────────────────────────────────────────

def differentiate(expr: str = None, variable: str = 'x', order: int = 1) -> Optional[str]:
    """
    式を変数で微分する。

    例:
      differentiate("sin(x)")   → "cos(x)"
      differentiate("x^3")      → "3*x**2"
      differentiate("x^3", order=2) → "6*x"
    """
    import sympy
    try:
        if not expr:
            return None
        var = _get_var(variable)
        e = _parse_expr(expr)
        result = sympy.diff(e, var, order)
        simplified = sympy.simplify(result)
        # 整数・有理数なら数値で返す
        try:
            fv = float(simplified)
            return int(fv) if fv == int(fv) else round(fv, 6)
        except Exception:
            return str(simplified)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# 積分
# ─────────────────────────────────────────────────────────────────────

def integrate_expr(expr: str, variable: str = 'x') -> Optional[str]:
    """
    式を不定積分する（+C は省略）。

    例: integrate_expr("2*x") → "x**2"
    """
    import sympy
    try:
        var = _get_var(variable)
        e = _parse_expr(expr)
        result = sympy.integrate(e, var)
        return str(sympy.simplify(result))
    except Exception:
        return None


def integrate_definite(expr: str = None, variable: str = None, lo: float = None, hi: float = None) -> Optional[float]:
    """
    定積分を計算する。

    例: integrate_definite("x^2", "x", 0, 1) → 0.333...
    """
    import sympy
    try:
        if not expr or variable is None or lo is None or hi is None:
            return None
        var = _get_var(variable)
        e = _parse_expr(expr)
        result = sympy.integrate(e, (var, lo, hi))
        fv = float(result.evalf())
        return round(fv, 8)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# 極限
# ─────────────────────────────────────────────────────────────────────

def compute_limit(expr: str, variable: str, point: str) -> Optional[Any]:
    """
    極限値を計算する。

    例:
      compute_limit("sin(x)/x", "x", "0") → 1
      compute_limit("1/x", "x", "oo")     → 0
    """
    import sympy
    try:
        var = _get_var(variable)
        e = _parse_expr(expr)
        pt = sympy.sympify(point.replace('oo', 'oo').replace('inf', 'oo'))
        result = sympy.limit(e, var, pt)
        try:
            fv = float(result.evalf())
            return int(fv) if fv == int(fv) else round(fv, 8)
        except Exception:
            return str(result)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# 式の評価・簡略化
# ─────────────────────────────────────────────────────────────────────

def evaluate_expr(expr: str) -> Optional[Any]:
    """
    数式文字列を評価して数値を返す。

    例: evaluate_expr("2^10") → 1024
    """
    import sympy
    try:
        e = _parse_expr(expr)
        result = e.evalf()
        fv = float(result)
        return int(fv) if fv == int(fv) else round(fv, 8)
    except Exception:
        return None


def simplify_expression(expr: str) -> Optional[str]:
    """
    代数式を簡略化する。

    例: simplify_expression("x^2 - 1") → "(x - 1)*(x + 1)"
    """
    import sympy
    try:
        e = _parse_expr(expr)
        result = sympy.factor(e)
        return str(result)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# 簡易テスト
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("solve_equation", solve_equation("x^2 - 5*x + 6 = 0"), [2, 3]),
        ("solve_linear",   solve_linear(2, 3, 7), 2),
        ("solve_quadratic",solve_quadratic(1, -5, 6), [2, 3]),
        ("differentiate",  differentiate("x^3"), "3*x**2"),
        ("integrate_expr", integrate_expr("2*x"), "x**2"),
        ("integrate_def",  integrate_definite("x^2", "x", 0, 1), 0.33333333),
        ("compute_limit",  compute_limit("sin(x)/x", "x", "0"), 1),
        ("evaluate_expr",  evaluate_expr("2^10"), 1024),
        ("simplify_expr",  simplify_expression("x^2 - 1"), "(x - 1)*(x + 1)"),
    ]
    for name, got, expected in tests:
        ok = "✅" if str(got) == str(expected) or got == expected else "⚠️ "
        print(f"{ok} {name}: {got!r}  (expected: {expected!r})")
