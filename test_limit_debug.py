"""
Limit Debug Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

import re
from sympy import symbols, limit as sympy_limit
from sympy.parsing.sympy_parser import parse_expr

# ヘルパー関数をコピー
def _parse_expression(text: str):
    text_lower = text.lower()
    
    # "of ..." パターン
    if " of " in text_lower:
        after_of = text[text_lower.index(" of ") + 4:]
        # "with respect to"の前まで
        if "with respect" in after_of.lower():
            expr = after_of[:after_of.lower().index("with respect")].strip()
        # "as"の前まで（極限の場合）
        elif " as " in after_of.lower():
            expr = after_of[:after_of.lower().index(" as ")].strip()
        else:
            expr = after_of.strip().rstrip(".")
        return expr
    
    return None

# テスト
source_text = "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2."
print("Source text:", source_text)

expr_str = _parse_expression(source_text)
print("Parsed expression:", expr_str)

# ポイント抽出
match = re.search(r'approaches\s+(\d+)', source_text.lower())
if match:
    point = float(match.group(1))
    print("Point:", point)
else:
    print("Point: NOT FOUND")

if expr_str and point:
    try:
        expr_str = expr_str.replace("^", "**")
        print("Transformed expr:", expr_str)
        
        var = symbols('x')
        expr = parse_expr(expr_str)
        print("Sympy expr:", expr)
        
        lim = sympy_limit(expr, var, point)
        print("Limit:", lim)
    except Exception as e:
        print("Error:", e)
