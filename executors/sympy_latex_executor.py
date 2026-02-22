"""
sympy_latex_executor.py
LaTeX数式 → SymPy → 計算 → 答え

HLE の数学問題の多くは LaTeX 形式。これを SymPy に変換して直接計算する。

対象:
  - 代数方程式: "Solve x^2 + 3x - 4 = 0"
  - 積分: "\\int_0^1 x^2 dx"
  - 極限: "\\lim_{x \\to 0} \\frac{\\sin x}{x}"
  - 和: "\\sum_{k=1}^{10} k^2"
  - 行列式: "\\det \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"
  - 数値評価: "2^{100} \\mod 7"

鉄の壁: 問題文から数式部分だけを抽出して計算。LLMは使わない。
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, Any


def extract_and_solve_latex(problem_text: str) -> Optional[Tuple[str, float, str]]:
    """
    問題文から LaTeX 数式を抽出し、SymPy で計算する。

    Returns:
        (answer_str, confidence, method) or None
    """
    # LaTeX 式を抽出
    expressions = _extract_latex_expressions(problem_text)
    if not expressions:
        return None

    # 問題の意図を推定
    intent = _detect_intent(problem_text)

    for expr_str in expressions:
        result = _solve_latex_expression(expr_str, intent, problem_text)
        if result is not None:
            answer_str = _format_answer(result)
            if answer_str:
                return answer_str, 0.85, f"sympy_latex:{intent}"

    return None


def _extract_latex_expressions(text: str) -> list[str]:
    """テキストから LaTeX 数式を抽出"""
    expressions = []

    # $...$ or $$...$$ 
    for m in re.finditer(r'\$\$([^$]+)\$\$|\$([^$]+)\$', text):
        expr = m.group(1) or m.group(2)
        if expr and len(expr.strip()) > 2:
            expressions.append(expr.strip())

    # \[...\] display math
    for m in re.finditer(r'\\\[(.+?)\\\]', text, re.DOTALL):
        expressions.append(m.group(1).strip())

    # \(...\) inline math
    for m in re.finditer(r'\\\((.+?)\\\)', text, re.DOTALL):
        expressions.append(m.group(1).strip())

    # Plain equation patterns: "x^2 + 3x - 4 = 0"
    for m in re.finditer(r'([a-z]\^?\d?\s*[+\-*/]\s*[\da-z^+\-*/\s()]+\s*=\s*[\d\-]+)', text):
        expressions.append(m.group(1).strip())

    return expressions


def _detect_intent(text: str) -> str:
    """問題の意図を推定"""
    text_lower = text.lower()

    if any(w in text_lower for w in ['solve', 'find the value', 'find x', 'find the root']):
        return 'solve'
    if any(w in text_lower for w in ['integral', 'integrate', '∫', 'int_', 'int{']):
        return 'integrate'
    if any(w in text_lower for w in ['derivative', 'differentiate', "d/dx", 'diff']):
        return 'differentiate'
    if any(w in text_lower for w in ['limit', 'lim_', 'lim{']):
        return 'limit'
    if any(w in text_lower for w in ['sum', 'summation', 'sigma', 'sum_', 'sum{']):
        return 'sum'
    if any(w in text_lower for w in ['determinant', 'det']):
        return 'determinant'
    if any(w in text_lower for w in ['modulo', 'mod ', 'remainder']):
        return 'modular'
    if any(w in text_lower for w in ['simplify', 'reduce', 'express']):
        return 'simplify'
    if any(w in text_lower for w in ['evaluate', 'compute', 'calculate', 'what is']):
        return 'evaluate'

    return 'evaluate'


def _latex_to_sympy(latex_str: str) -> Optional[Any]:
    """LaTeX 文字列を SymPy 式に変換"""
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex_str)
    except Exception:
        pass

    # フォールバック: 手動変換
    try:
        import sympy
        expr_str = latex_str

        # LaTeX → Python 変換
        expr_str = expr_str.replace('\\cdot', '*')
        expr_str = expr_str.replace('\\times', '*')
        expr_str = expr_str.replace('\\div', '/')
        expr_str = expr_str.replace('\\pm', '+')
        expr_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr_str)
        expr_str = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', expr_str)
        expr_str = re.sub(r'\\sqrt\[(\d+)\]\{([^}]+)\}', r'(\2)**(1/\1)', expr_str)
        expr_str = expr_str.replace('^', '**')
        expr_str = re.sub(r'\\left|\\right', '', expr_str)
        expr_str = re.sub(r'\\[a-zA-Z]+', '', expr_str)  # 残りのコマンド除去
        expr_str = re.sub(r'\{|\}', '', expr_str)  # ブレース除去

        # 暗黙の乗算
        expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
        expr_str = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr_str)

        from sympy.abc import x, y, z, n, k, t, a, b, c
        local_ns = {
            'x': x, 'y': y, 'z': z, 'n': n, 'k': k, 't': t,
            'a': a, 'b': b, 'c': c,
            'pi': sympy.pi, 'e': sympy.E, 'i': sympy.I,
            'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
            'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.log,
            'sqrt': sympy.sqrt, 'abs': sympy.Abs,
            'factorial': sympy.factorial,
        }
        return sympy.sympify(expr_str, locals=local_ns)
    except Exception:
        return None


def _solve_latex_expression(
    latex_str: str, intent: str, full_text: str
) -> Optional[Any]:
    """LaTeX 式を意図に応じて計算"""
    try:
        import sympy
        from sympy.abc import x, y, z, n, k

        expr = _latex_to_sympy(latex_str)
        if expr is None:
            return None

        if intent == 'solve':
            # 方程式を解く
            if '=' in latex_str:
                parts = latex_str.split('=')
                lhs = _latex_to_sympy(parts[0].strip())
                rhs = _latex_to_sympy(parts[1].strip())
                if lhs is not None and rhs is not None:
                    solutions = sympy.solve(lhs - rhs, x)
                    if solutions:
                        return solutions if len(solutions) > 1 else solutions[0]
            else:
                solutions = sympy.solve(expr, x)
                if solutions:
                    return solutions if len(solutions) > 1 else solutions[0]

        elif intent == 'integrate':
            # 積分
            bounds = _extract_bounds(full_text)
            if bounds:
                a, b = bounds
                result = sympy.integrate(expr, (x, a, b))
            else:
                result = sympy.integrate(expr, x)
            if result is not None and result != sympy.S.NaN:
                return result

        elif intent == 'differentiate':
            return sympy.diff(expr, x)

        elif intent == 'limit':
            # 極限点を検出
            limit_point = _extract_limit_point(full_text)
            if limit_point is not None:
                return sympy.limit(expr, x, limit_point)

        elif intent == 'sum':
            bounds = _extract_sum_bounds(full_text)
            if bounds:
                lower, upper = bounds
                return sympy.summation(expr, (k, lower, upper))

        elif intent == 'modular':
            mod_val = _extract_mod_value(full_text)
            if mod_val:
                result = expr.evalf()
                if result.is_integer:
                    return int(result) % mod_val

        elif intent == 'simplify':
            return sympy.simplify(expr)

        elif intent == 'evaluate':
            result = expr.evalf()
            if result.is_number:
                # 整数なら整数で返す
                if result == int(result):
                    return int(result)
                return result

    except Exception:
        pass

    return None


def _extract_bounds(text: str) -> Optional[tuple]:
    """積分の上下限を抽出"""
    import sympy
    # \int_a^b or \int_{a}^{b}
    m = re.search(r'\\int_\{?([^{}]+?)\}?\^\{?([^{}]+?)\}?[\s\\]', text)
    if m:
        try:
            a = sympy.sympify(m.group(1).strip())
            b = sympy.sympify(m.group(2).strip())
            return (a, b)
        except Exception:
            pass
    # "from a to b"
    m = re.search(r'from\s+([0-9.eπ]+)\s+to\s+([0-9.eπ∞]+)', text, re.IGNORECASE)
    if m:
        try:
            a_str = m.group(1).replace('π', 'pi').replace('∞', 'oo')
            b_str = m.group(2).replace('π', 'pi').replace('∞', 'oo')
            a = sympy.sympify(a_str)
            b = sympy.sympify(b_str)
            return (a, b)
        except Exception:
            pass
    return None


def _extract_limit_point(text: str) -> Optional[Any]:
    """極限の点を抽出"""
    import sympy
    m = re.search(r'\\to\s+([0-9.∞]+|\\infty)', text)
    if m:
        val = m.group(1)
        if val in ('∞', '\\infty'):
            return sympy.oo
        try:
            return sympy.sympify(val)
        except Exception:
            pass
    # "as x approaches ..."
    m = re.search(r'approaches?\s+([0-9.]+|infinity)', text, re.IGNORECASE)
    if m:
        val = m.group(1)
        if val.lower() == 'infinity':
            return sympy.oo
        return sympy.sympify(val)
    return None


def _extract_sum_bounds(text: str) -> Optional[tuple]:
    """和の上下限を抽出"""
    import sympy
    m = re.search(r'\\sum_\{?[a-z]\s*=\s*(\d+)\}?\^\{?(\d+|\\infty|n)\}?', text)
    if m:
        lower = int(m.group(1))
        upper_str = m.group(2)
        if upper_str in ('\\infty', 'n'):
            return (lower, sympy.oo)
        return (lower, int(upper_str))
    # "for k = 1 to N"
    m = re.search(r'for\s+[a-z]\s*=\s*(\d+)\s+to\s+(\d+)', text, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _extract_mod_value(text: str) -> Optional[int]:
    """mod の値を抽出"""
    m = re.search(r'(?:mod|modulo)\s+(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\\pmod\{(\d+)\}', text)
    if m:
        return int(m.group(1))
    return None


def _format_answer(result: Any) -> Optional[str]:
    """SymPy結果を文字列に変換"""
    if result is None:
        return None
    import sympy

    if isinstance(result, (int, float)):
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)

    if isinstance(result, list):
        return ", ".join(str(r) for r in result)

    if hasattr(result, 'is_number') and result.is_number:
        evaled = result.evalf()
        if evaled == int(evaled):
            return str(int(evaled))
        # 有理数ならそのまま
        if result.is_rational:
            return str(result)
        return str(evaled)

    # SymPy 式をそのまま文字列化
    s = str(result)
    if s and s != 'None' and s != 'nan':
        return s
    return None
