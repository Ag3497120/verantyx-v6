#!/usr/bin/env python3
"""
Equation Solver & Answer Matcherのテスト
"""
from executors.equation_solver import solve_linear_equation, solve_quadratic_equation, algebra_solve_equation
from core.answer_matcher import flexible_match, normalize_number

print("=" * 70)
print("Equation Solver & Answer Matcher Test")
print("=" * 70)

# Test 1: Linear equations
print("\n[Test 1] Linear Equations")
test_cases = [
    ("2x + 3 = 11", 4),
    ("x + 5 = 10", 5),
    ("3x = 15", 5),
    ("2x - 4 = 10", 7),
]

for eq, expected in test_cases:
    result = solve_linear_equation(eq)
    if result.get('success'):
        solution = result['solution']
        match = abs(solution - expected) < 1e-6
        status = "✓" if match else "✗"
        print(f"  {status} {eq} → x = {solution} (expected: {expected})")
    else:
        print(f"  ✗ {eq} → ERROR: {result.get('error')}")

# Test 2: Quadratic equations
print("\n[Test 2] Quadratic Equations")
quad_tests = [
    {"a": 1, "b": -5, "c": 6, "expected": [3, 2]},  # x² - 5x + 6 = 0 → x = 2, 3
    {"a": 1, "b": 0, "c": -4, "expected": [2, -2]},  # x² - 4 = 0 → x = ±2
    {"a": 1, "b": -2, "c": 1, "expected": [1]},  # x² - 2x + 1 = 0 → x = 1 (repeated)
]

for test in quad_tests:
    result = solve_quadratic_equation(test['a'], test['b'], test['c'])
    if result.get('success'):
        solutions = result['solutions']
        solutions_sorted = sorted([float(s) for s in solutions if isinstance(s, (int, float))])
        expected_sorted = sorted(test['expected'])
        match = all(abs(s - e) < 1e-6 for s, e in zip(solutions_sorted, expected_sorted))
        status = "✓" if match else "✗"
        print(f"  {status} {test['a']}x² + {test['b']}x + {test['c']} = 0 → {solutions} (expected: {test['expected']})")
    else:
        print(f"  ✗ ERROR: {result.get('error')}")

# Test 3: Answer Matcher
print("\n[Test 3] Answer Matcher")
match_tests = [
    ("4", "4.0", True),
    ("4.000", 4, True),
    ("  4  ", "4", True),
    ("3.14159", "3.14160", True),  # 許容誤差内
    ("Z+Z+Z", "z+z+z", True),  # 正規化
    ("\\mathbb{Z}", "z", True),  # LaTeX正規化
    ("3+4i", complex(3, 4), True),  # 複素数
    ("5", "6", False),
]

for pred, exp, expected_match in match_tests:
    result = flexible_match(pred, exp)
    status = "✓" if result == expected_match else "✗"
    print(f"  {status} flexible_match({repr(pred)}, {repr(exp)}) = {result} (expected: {expected_match})")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
