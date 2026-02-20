"""Test executor parameter passing"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.algebra import factor_polynomial, solve_linear_equation, partition_number

# Test factor_polynomial
print("Testing factor_polynomial:")
try:
    result = factor_polynomial()  # No args
    print(f"  No args: {result}")
except TypeError as e:
    print(f"  No args: ERROR - {e}")

try:
    result = factor_polynomial("x^2 - 5*x + 6")
    print(f"  With expression: {result}")
except Exception as e:
    print(f"  With expression: ERROR - {e}")

# Test solve_linear_equation
print("\nTesting solve_linear_equation:")
try:
    result = solve_linear_equation()
    print(f"  No args: {result}")
except TypeError as e:
    print(f"  No args: ERROR - {e}")

try:
    result = solve_linear_equation("2*x + 4 = 0")
    print(f"  With equation: {result}")
except Exception as e:
    print(f"  With equation: ERROR - {e}")

# Test partition_number
print("\nTesting partition_number:")
try:
    result = partition_number()
    print(f"  No args: {result}")
except TypeError as e:
    print(f"  No args: ERROR - {e}")

try:
    result = partition_number(5)
    print(f"  With n=5: {result}")
except Exception as e:
    print(f"  With n=5: ERROR - {e}")
