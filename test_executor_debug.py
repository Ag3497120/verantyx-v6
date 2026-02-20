"""Quick test of the three regressed executors"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.algebra import solve_linear_equation, factor_polynomial, partition_number

print("="*80)
print("Testing Regressed Executors")
print("="*80)

# Test solve_linear_equation
print("\n[solve_linear_equation]")
test_cases = [
    ("2x + 4 = 0", -2.0),
    ("3x = 9", 3.0),
    ("x + 5 = 10", 5.0),
    ("2*x - 6 = 0", 3.0),
]
for eq, expected in test_cases:
    result = solve_linear_equation(eq)
    status = "✅" if result is not None and abs(result - expected) < 1e-6 else "❌"
    print(f"{status} solve_linear_equation('{eq}') = {result} (expected: {expected})")

# Test factor_polynomial
print("\n[factor_polynomial]")
factor_cases = [
    ("x^2 - 5*x + 6", "(x - 3)*(x - 2)"),  # approximate expected
    ("x^2 - 1", "(x - 1)*(x + 1)"),
]
for expr, expected_pattern in factor_cases:
    result = factor_polynomial(expr)
    print(f"  factor_polynomial('{expr}') = {result}")
    print(f"    Expected pattern: {expected_pattern}")

# Test partition_number
print("\n[partition_number]")
partition_cases = [
    (0, 1),
    (1, 1),
    (2, 2),  # 2 = 2, 1+1
    (3, 3),  # 3 = 3, 2+1, 1+1+1
    (4, 5),  # 4 = 4, 3+1, 2+2, 2+1+1, 1+1+1+1
    (5, 7),
    (10, 42),
]
for n, expected in partition_cases:
    result = partition_number(n)
    status = "✅" if result == expected else "❌"
    print(f"{status} partition_number({n}) = {result} (expected: {expected})")

print("\n" + "="*80)
