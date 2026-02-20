"""
IR Structure Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer
import json

question = "Find the determinant of matrix [[2, 3], [1, 4]]."

decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()

print("IR dict keys:", list(ir_dict.keys()))
print()
print("source_text in IR?", "source_text" in ir_dict)
print("source_text value:", ir_dict.get("source_text", "NOT FOUND"))
print()
print("metadata:", ir_dict.get("metadata"))
print()
print("metadata.source_text:", ir_dict.get("metadata", {}).get("source_text", "NOT FOUND"))
