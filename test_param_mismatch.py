"""Test if parameter mismatch is causing wrong executor results"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.algebra import solve_linear_equation, factor_polynomial, partition_number

# These functions expect specific types of input
# Let's test what happens with mismatched inputs

print("="*80)
print("TESTING PARAMETER MISMATCHES")
print("="*80)

# Test 1: What if solve_linear_equation gets a number instead of string?
print("\n[Test 1: solve_linear_equation with wrong types]")
test_inputs = [
    None,
    123,
    {"equation": "x + 5 = 10"},
    "not an equation",
    "",
]
for inp in test_inputs:
    try:
        result = solve_linear_equation(inp)
        print(f"  solve_linear_equation({repr(inp)}) = {result}")
    except Exception as e:
        print(f"  solve_linear_equation({repr(inp)}) ERROR: {e}")

# Test 2: factor_polynomial with wrong types
print("\n[Test 2: factor_polynomial with wrong types]")
test_inputs2 = [
    None,
    456,
    {"expr": "x^2 - 1"},
    "",
]
for inp in test_inputs2:
    try:
        result = factor_polynomial(inp)
        print(f"  factor_polynomial({repr(inp)}) = {result}")
    except Exception as e:
        print(f"  factor_polynomial({repr(inp)}) ERROR: {e}")

# Test 3: partition_number with wrong types
print("\n[Test 3: partition_number with wrong types]")
test_inputs3 = [
    None,
    "5",
    {"n": 5},
    -1,
    1000,
]
for inp in test_inputs3:
    try:
        result = partition_number(inp)
        print(f"  partition_number({repr(inp)}) = {result}")
    except Exception as e:
        print(f"  partition_number({repr(inp)}) ERROR: {e}")

print("\n" + "="*80)
print("HYPOTHESIS: If executors return None for bad inputs,")
print("then the issue is NOT parameter mismatch.")
print("="*80)
