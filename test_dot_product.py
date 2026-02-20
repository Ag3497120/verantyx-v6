"""
Dot Product Debug
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.linear_algebra import dot_product

# Test case 1
question1 = "Calculate the dot product of vectors [1, 2, 3] and [4, 5, 6]."
ir1 = {"metadata": {"source_text": question1}}

print("Test 1:")
print(f"  Question: {question1}")
result1 = dot_product(ir=ir1)
print(f"  Result: {result1}")
print(f"  Expected: 32 (1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)")
print()

# Test case 2
question2 = "Find the dot product of [2, 0] and [0, 3]."
ir2 = {"metadata": {"source_text": question2}}

print("Test 2:")
print(f"  Question: {question2}")
result2 = dot_product(ir=ir2)
print(f"  Result: {result2}")
print(f"  Expected: 0 (2*0 + 0*3 = 0)")
