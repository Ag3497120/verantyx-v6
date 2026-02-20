"""
Debug multiple choice piece search
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pieces.piece import PieceDB
from decomposer.decomposer import RuleBasedDecomposer

# Test question
test_question = """Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?

Answer Choices:
A. Egalitarian Dominance
B. General Non-Extreme Priority
C. Non-Elitism
D. Weak Non-Sadism
E. Weak Quality Addition"""

print("Debug: Multiple Choice Piece Search")
print("=" * 80)

# Get IR
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(test_question)
ir_dict = ir.to_dict()

print("\nIR:")
print(f"  domain: {ir_dict['domain']}")
print(f"  task: {ir_dict['task']}")
print(f"  answer_schema: {ir_dict['answer_schema']}")

# Load piece DB
piece_db = PieceDB("pieces/piece_db.jsonl")

# Find solve_multiple_choice piece
print("\nsolve_multiple_choice piece:")
mc_piece = piece_db.find_by_id("solve_multiple_choice")
if mc_piece:
    print(f"  piece_id: {mc_piece.piece_id}")
    print(f"  requires: {mc_piece.in_spec.requires}")
    print(f"  confidence: {mc_piece.confidence}")
    print(f"  out.schema: {mc_piece.out_spec.schema}")
    
    # Test match score
    match_score = mc_piece.matches_ir(ir_dict)
    print(f"\n  Match score: {match_score}")
    
    # Manual check
    print("\n  Manual check:")
    for req in mc_piece.in_spec.requires:
        if ":" in req:
            req_type, req_value = req.split(":", 1)
            ir_value = ir_dict.get(req_type)
            matches = ir_value == req_value
            print(f"    {req_type}:{req_value} vs IR {req_type}:{ir_value} = {matches}")
else:
    print("  NOT FOUND in piece DB!")

# Search all pieces
print("\nPiece DB search results:")
pieces = piece_db.search(ir_dict, top_k=10)
print(f"  Found {len(pieces)} pieces")
for i, (piece, score) in enumerate(pieces[:5], 1):
    print(f"  {i}. {piece.piece_id} (score={score:.2f})")
    print(f"     requires: {piece.in_spec.requires}")

# Check if solve_multiple_choice is in the results
mc_in_results = any(p.piece_id == "solve_multiple_choice" for p, _ in pieces)
print(f"\n✓ solve_multiple_choice in results: {mc_in_results}")
