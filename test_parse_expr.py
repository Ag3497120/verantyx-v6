"""
Parse Expression Debug
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.calculus import _parse_expression, _parse_variable

test_cases = [
    "Find the derivative of 3x^3.",
    "What is the integral of 2x with respect to x?",
    "What is the derivative of 5x?",
    "What is the derivative of x^2 with respect to x?",  # This one works
]

for text in test_cases:
    expr = _parse_expression(text)
    var = _parse_variable(text)
    print(f"Text: {text}")
    print(f"  Expression: {expr}")
    print(f"  Variable: {var}")
    print()
