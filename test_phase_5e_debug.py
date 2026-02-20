"""
Phase 5E Debug Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

# テスト1: 行列式
question = "Find the determinant of matrix [[2, 3], [1, 4]]."

pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
result = pipeline.solve(question)

print("Question:", question)
print("Result:", result)
print()

# IRを確認
from decomposer.decomposer import RuleBasedDecomposer
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
print("IR:", ir.to_dict())
