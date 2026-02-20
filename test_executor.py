"""
Executor直接テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from executors.arithmetic import evaluate
from core.ir import IR, TaskType, Domain, AnswerSchema, Entity

# Test 1: 直接実行
print("=== Test 1: 直接実行 ===")
result = evaluate(expression="1+1")
print(f"Result: {result}")

# Test 2: IRから実行
print("\n=== Test 2: IRから実行 ===")
ir = IR(
    task=TaskType.COMPUTE,
    domain=Domain.ARITHMETIC,
    answer_schema=AnswerSchema.INTEGER,
    metadata={"source_text": "What is 1 + 1?"}
)
ir_dict = ir.to_dict()
print(f"IR: {ir_dict}")

result2 = evaluate(ir=ir_dict)
print(f"Result: {result2}")

# Test 3: 5 * 6
print("\n=== Test 3: 5 * 6 ===")
ir3 = IR(
    task=TaskType.COMPUTE,
    domain=Domain.ARITHMETIC,
    answer_schema=AnswerSchema.INTEGER,
    metadata={"source_text": "Calculate 5 * 6."}
)
result3 = evaluate(ir=ir3.to_dict())
print(f"Result: {result3}")
