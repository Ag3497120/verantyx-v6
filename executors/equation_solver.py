"""
Equation Solver - 方程式ソルバー

線形方程式、二次方程式、連立方程式の解法
"""
import re
from typing import Optional, List, Tuple, Dict, Any, Union


def solve_linear_equation(equation: str) -> Optional[float]:
    """
    線形方程式を解く（ax + b = c形式）
    
    Args:
        equation: 方程式文字列（例: "2x + 3 = 7", "x - 5 = 10"）
    
    Returns:
        解（x の値）、解けない場合None
    """
    try:
        # 方程式を左辺と右辺に分割
        if '=' not in equation:
            return None
        
        left, right = equation.split('=')
        left = left.strip()
        right = right.strip()
        
        # 右辺を数値に変換
        try:
            c = float(eval(right))
        except:
            return None
        
        # 左辺をパース（ax + b形式）
        # パターン: "2x + 3", "x - 5", "3x", "x"
        
        # xの係数と定数項を抽出
        a = 0.0  # xの係数
        b = 0.0  # 定数項
        
        # xを含む項と定数項を分離
        terms = re.split(r'([+-])', left)
        
        current_sign = 1
        for i, term in enumerate(terms):
            term = term.strip()
            
            if term == '+':
                current_sign = 1
            elif term == '-':
                current_sign = -1
            elif term:
                if 'x' in term:
                    # xを含む項
                    coef = term.replace('x', '').strip().rstrip('*').strip()
                    if coef == '' or coef == '+':
                        a += current_sign * 1
                    elif coef == '-':
                        a += current_sign * (-1)
                    else:
                        a += current_sign * float(coef)
                else:
                    # 定数項
                    b += current_sign * float(term)
        
        # ax + b = c → x = (c - b) / a
        if abs(a) < 1e-10:
            return None  # xの係数が0の場合
        
        x = (c - b) / a
        return x
        
    except Exception as e:
        return None


def solve_quadratic_equation(a: float, b: float, c: float) -> List[complex]:
    """
    二次方程式 ax^2 + bx + c = 0 を解く
    
    Args:
        a: x^2の係数
        b: xの係数
        c: 定数項
    
    Returns:
        解のリスト（実数または複素数）
    """
    if abs(a) < 1e-10:
        # 線形方程式に退化
        if abs(b) < 1e-10:
            return []
        return [complex(-c / b, 0)]
    
    # 判別式
    discriminant = b * b - 4 * a * c
    
    if discriminant >= 0:
        # 実数解
        sqrt_d = discriminant ** 0.5
        x1 = (-b + sqrt_d) / (2 * a)
        x2 = (-b - sqrt_d) / (2 * a)
        return [complex(x1, 0), complex(x2, 0)]
    else:
        # 複素数解
        real_part = -b / (2 * a)
        imag_part = (abs(discriminant) ** 0.5) / (2 * a)
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]


def parse_quadratic_equation(equation: str) -> Optional[Tuple[float, float, float]]:
    """
    二次方程式の文字列から係数を抽出
    
    Args:
        equation: 方程式文字列（例: "x^2 + 2x + 1 = 0"）
    
    Returns:
        (a, b, c) のタプル、パース失敗時None
    """
    try:
        # = 0 の形式に正規化
        if '=' in equation:
            left, right = equation.split('=')
            left = left.strip()
            right = right.strip()
            
            # 右辺を左辺に移項
            if right != '0':
                left = f"({left}) - ({right})"
        else:
            left = equation
        
        # x^2, x, 定数項の係数を抽出
        a = 0.0
        b = 0.0
        c = 0.0
        
        # x^2の係数
        match = re.search(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*x\s*\^\s*2', left)
        if match:
            coef = match.group(1).replace(' ', '')
            if coef in ['', '+']:
                a = 1.0
            elif coef == '-':
                a = -1.0
            else:
                a = float(coef)
        
        # xの係数（x^2ではない）
        # x^2の項を除去してからxを探す
        temp = re.sub(r'[+-]?\s*\d*\.?\d*\s*\*?\s*x\s*\^\s*2', '', left)
        match = re.search(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*x(?!\^)', temp)
        if match:
            coef = match.group(1).replace(' ', '')
            if coef in ['', '+']:
                b = 1.0
            elif coef == '-':
                b = -1.0
            else:
                b = float(coef)
        
        # 定数項（xを含まない項）
        temp = re.sub(r'[+-]?\s*\d*\.?\d*\s*\*?\s*x\s*\^\s*2', '', left)
        temp = re.sub(r'[+-]?\s*\d*\.?\d*\s*\*?\s*x(?!\^)', '', temp)
        temp = temp.strip()
        if temp:
            c = float(eval(temp))
        
        return (a, b, c)
        
    except Exception as e:
        return None


def solve_system_linear(equations: List[str]) -> Optional[Dict[str, float]]:
    """
    連立一次方程式を解く（2変数のみ）
    
    Args:
        equations: 方程式のリスト（例: ["2x + 3y = 7", "x - y = 1"]）
    
    Returns:
        {"x": value, "y": value} の辞書、解けない場合None
    """
    if len(equations) != 2:
        return None
    
    try:
        # 各方程式を ax + by = c 形式にパース
        coeffs = []
        for eq in equations:
            if '=' not in eq:
                return None
            
            left, right = eq.split('=')
            c = float(eval(right.strip()))
            
            # xの係数
            a = 0.0
            match = re.search(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*x', left)
            if match:
                coef = match.group(1).replace(' ', '')
                if coef in ['', '+']:
                    a = 1.0
                elif coef == '-':
                    a = -1.0
                else:
                    a = float(coef)
            
            # yの係数
            b = 0.0
            match = re.search(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*y', left)
            if match:
                coef = match.group(1).replace(' ', '')
                if coef in ['', '+']:
                    b = 1.0
                elif coef == '-':
                    b = -1.0
                else:
                    b = float(coef)
            
            coeffs.append((a, b, c))
        
        # クラメルの公式で解く
        a1, b1, c1 = coeffs[0]
        a2, b2, c2 = coeffs[1]
        
        # 行列式
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:
            return None  # 解が存在しない or 無数に存在
        
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return {"x": x, "y": y}
        
    except Exception as e:
        return None


def main():
    """テスト実行"""
    print("=" * 80)
    print("Equation Solver Test")
    print("=" * 80)
    print()
    
    # 線形方程式テスト
    print("[Linear Equations]")
    linear_tests = [
        ("2x + 3 = 7", 2.0),
        ("x - 5 = 10", 15.0),
        ("3x = 12", 4.0),
        ("x = 5", 5.0),
    ]
    
    for eq, expected in linear_tests:
        result = solve_linear_equation(eq)
        status = "✅" if result is not None and abs(result - expected) < 1e-6 else "❌"
        print(f"{status} solve_linear_equation('{eq}') = {result} (expected: {expected})")
    
    print()
    
    # 二次方程式テスト
    print("[Quadratic Equations]")
    quad_tests = [
        ((1, -3, 2), [2, 1]),  # x^2 - 3x + 2 = 0
        ((1, 0, -4), [2, -2]),  # x^2 - 4 = 0
        ((1, -2, 1), [1, 1]),  # x^2 - 2x + 1 = 0
    ]
    
    for (a, b, c), expected_real in quad_tests:
        solutions = solve_quadratic_equation(a, b, c)
        real_solutions = sorted([s.real for s in solutions if abs(s.imag) < 1e-6])
        status = "✅" if len(real_solutions) == len(expected_real) else "❌"
        print(f"{status} solve_quadratic({a}, {b}, {c}) = {real_solutions} (expected: {expected_real})")
    
    print()
    
    # 連立方程式テスト
    print("[System of Linear Equations]")
    system_tests = [
        (["2x + 3y = 7", "x - y = 1"], {"x": 2.0, "y": 1.0}),
        (["x + y = 5", "x - y = 1"], {"x": 3.0, "y": 2.0}),
    ]
    
    for eqs, expected in system_tests:
        result = solve_system_linear(eqs)
        if result:
            match = all(abs(result.get(k, 0) - v) < 1e-6 for k, v in expected.items())
            status = "✅" if match else "❌"
        else:
            status = "❌"
        print(f"{status} solve_system({eqs}) = {result} (expected: {expected})")
    
    print()
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
