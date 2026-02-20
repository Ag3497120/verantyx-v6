"""
Decomposerテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from decomposer.decomposer import RuleBasedDecomposer

decomposer = RuleBasedDecomposer()

# Test 1
text = "What is 1 + 1?"
print(f"Text: {text}")
print()

ir = decomposer.decompose(text)
ir_dict = ir.to_dict()

print("IR:")
print(f"  task: {ir_dict['task']}")
print(f"  domain: {ir_dict['domain']}")
print(f"  answer_schema: {ir_dict['answer_schema']}")
print(f"  entities: {ir_dict['entities']}")
