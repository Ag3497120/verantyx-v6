"""
Limit IR Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer

question = "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2."

decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()

print(f"Question: {question}")
print(f"Domain: {ir_dict['domain']}")
print(f"Task: {ir_dict['task']}")
print(f"Answer Schema: {ir_dict['answer_schema']}")
print(f"Keywords: {ir_dict['metadata']['keywords']}")
