"""
Pipeline Detail Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB
from assembler.greedy_assembler import GreedyAssembler
from assembler.executor import Executor

# 問題文
question = "Find the determinant of matrix [[2, 3], [1, 4]]."

# Step 1: Decompose
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()
print("IR:", ir_dict)
print()

# Step 2: Retrieve pieces
piece_db = PieceDB()
scored_pieces = piece_db.retrieve(ir_dict, top_k=5)
print(f"Top pieces ({len(scored_pieces)}):")
for piece, score in scored_pieces:
    print(f"  {piece.piece_id}: {score:.3f} (requires={piece.in_spec.requires})")
print()

# Step 3: Assemble
assembler = GreedyAssembler()
path = assembler.assemble(ir_dict, scored_pieces)
if path:
    print(f"Selected path ({len(path)} pieces):")
    for piece in path:
        print(f"  {piece.piece_id}")
    print()
    
    # Step 4: Execute
    executor = Executor()
    candidate = executor.execute_path(path, ir_dict)
    print("Candidate:", candidate.to_dict() if candidate else None)
else:
    print("No path found")
