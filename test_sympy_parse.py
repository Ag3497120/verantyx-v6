"""
Sympy Parse Test
"""
from sympy import symbols, diff
from sympy.parsing.sympy_parser import parse_expr

test_cases = [
    "3x^3",
    "3x**3",
    "2x",
    "2*x",
    "5x",
    "5*x",
    "x^2",
    "x**2"
]

x = symbols('x')

for expr_str in test_cases:
    print(f"Expression: '{expr_str}'")
    try:
        # ^ を ** に変換
        expr_str_fixed = expr_str.replace("^", "**")
        print(f"  Fixed: '{expr_str_fixed}'")
        
        expr = parse_expr(expr_str_fixed)
        print(f"  Parsed: {expr}")
        
        derivative = diff(expr, x)
        print(f"  Derivative: {derivative}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
