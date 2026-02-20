"""
Limit Pipeline Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

question = "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2."

pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
result = pipeline.solve(question)

print(f"Question: {question}")
print(f"Answer: {result.get('answer')}")
print(f"Expected: 4")
print(f"Pieces: {result.get('pieces')}")
print(f"Confidence: {result.get('confidence')}")
print(f"Status: {result.get('status')}")
