"""
Single Case Test - Pipeline vs Direct Executor
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from executors.linear_algebra import matrix_determinant

# Test case
question = "Find the determinant of matrix [[2, 3], [1, 4]]."

# Pipeline test
print("=" * 80)
print("Pipeline Test")
print("=" * 80)
pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
result = pipeline.solve(question)
print(f"Question: {question}")
print(f"Answer: {result.get('answer')}")
print(f"Expected: 5")
print(f"Pieces: {result.get('pieces')}")
print(f"Confidence: {result.get('confidence')}")
print()

# Direct executor test
print("=" * 80)
print("Direct Executor Test")
print("=" * 80)
ir = {"source_text": question}
direct_result = matrix_determinant(ir=ir)
print(f"Result: {direct_result}")
print()

# Check piece selection
from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB

decomposer = RuleBasedDecomposer()
ir_obj = decomposer.decompose(question)
ir_dict = ir_obj.to_dict()

print("=" * 80)
print("IR & Piece Selection")
print("=" * 80)
print(f"Domain: {ir_dict['domain']}")
print(f"Task: {ir_dict['task']}")
print(f"Keywords: {ir_dict['metadata']['keywords']}")
print()

piece_db = PieceDB()
scored = piece_db.retrieve(ir_dict, top_k=5)
print("Top 5 pieces:")
for p, score in scored:
    print(f"  {score:.3f} - {p.piece_id} (requires: {p.in_spec.requires}, tags: {p.tags})")
