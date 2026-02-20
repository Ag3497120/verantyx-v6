"""
Simple Executor Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

# 微積分テスト
from executors.calculus import derivative

source_text = "What is the derivative of x^2 with respect to x?"
ir = {"source_text": source_text}

result = derivative(ir=ir)
print("Derivative test:")
print("  Question:", source_text)
print("  Result:", result)
print("  Expected: 2x or 2*x")
print()

# 極限テスト
from executors.calculus import limit

source_text2 = "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2."
ir2 = {"source_text": source_text2}

result2 = limit(ir=ir2)
print("Limit test:")
print("  Question:", source_text2)
print("  Result:", result2)
print("  Expected: 4")
