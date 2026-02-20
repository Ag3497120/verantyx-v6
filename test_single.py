"""
単一問題デバッグテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced

pipeline = VerantyxV6Enhanced()

# Test 1
problem = {
    "question": "What is 1 + 1?",
    "answer": "2"
}

print("Question:", problem["question"])
print("Expected:", problem["answer"])
print()

result = pipeline.solve(problem["question"], expected_answer=problem["answer"])

print("\n=== Result ===")
print("Answer:", result["answer"])
print("Status:", result["status"])
print("\n=== Trace ===")
for step in result["trace"]:
    print(f"  {step}")
