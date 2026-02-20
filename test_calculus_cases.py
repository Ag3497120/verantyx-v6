"""
Calculus Cases Debug
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.calculus import derivative, integral, limit

# Test 1: derivative 3x^3
q1 = "Find the derivative of 3x^3."
ir1 = {"metadata": {"source_text": q1}}
print("Test 1: derivative 3x^3")
print(f"  Question: {q1}")
result1 = derivative(ir=ir1)
print(f"  Result: {result1}")
print(f"  Expected: 9x^2")
print()

# Test 2: integral 2x
q2 = "What is the integral of 2x with respect to x?"
ir2 = {"metadata": {"source_text": q2}}
print("Test 2: integral 2x")
print(f"  Question: {q2}")
result2 = integral(ir=ir2)
print(f"  Result: {result2}")
print(f"  Expected: x^2")
print()

# Test 3: limit
q3 = "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2."
ir3 = {"metadata": {"source_text": q3}}
print("Test 3: limit")
print(f"  Question: {q3}")
result3 = limit(ir=ir3)
print(f"  Result: {result3}")
print(f"  Expected: 4")
print()

# Test 4: derivative 5x
q4 = "What is the derivative of 5x?"
ir4 = {"metadata": {"source_text": q4}}
print("Test 4: derivative 5x")
print(f"  Question: {q4}")
result4 = derivative(ir=ir4)
print(f"  Result: {result4}")
print(f"  Expected: 5")
