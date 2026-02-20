"""
Detailed debug of string_length execution
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from core.ir import IR
from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB
from assembler.beam_search import BeamSearch
from assembler.executor import Executor

# Test question
question = "What is the length of 'hello'?"
print(f"Question: {question}")
print("="*80)

# Step 1: Decompose
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()

print("\nStep 1: IR")
print(f"  Domain: {ir_dict['domain']}")
print(f"  Task: {ir_dict['task']}")
print(f"  Answer Schema: {ir_dict['answer_schema']}")
print(f"  Entities: {ir_dict.get('entities', [])}")
print(f"  Keywords: {ir_dict.get('metadata', {}).get('keywords', [])}")

# Step 2: Search pieces
piece_db = PieceDB("pieces/piece_db.jsonl")
pieces = piece_db.search(ir_dict, top_k=10)

print(f"\nStep 2: Piece Search (top {len(pieces)})")
for i, (piece, score) in enumerate(pieces[:5], 1):
    print(f"  {i}. {piece.piece_id} (score={score:.2f})")
    print(f"     executor: {piece.executor}")
    print(f"     requires: {piece.in_spec.requires}")
    print(f"     slots: {piece.in_spec.slots}")

# Step 3: Try beam search
assembler = BeamSearch(piece_db)
paths = assembler.find_paths(ir_dict, max_depth=3, beam_width=3)

print(f"\nStep 3: Beam Search ({len(paths)} paths)")
for i, (path, score) in enumerate(paths[:3], 1):
    piece_ids = [p.piece_id for p in path]
    print(f"  {i}. {' -> '.join(piece_ids)} (score={score:.2f})")

# Step 4: Execute
if paths:
    best_path, _ = paths[0]
    print(f"\nStep 4: Execute best path")
    print(f"  Path: {[p.piece_id for p in best_path]}")
    
    executor = Executor()
    result = executor.execute_path(best_path, ir_dict)
    
    if result:
        print(f"  Result: {result.to_dict()}")
    else:
        print(f"  Result: None (execution failed)")
else:
    print(f"\nStep 4: No paths found")
