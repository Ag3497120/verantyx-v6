"""
Debug string_length execution
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from pieces.piece import PieceDB

# Load piece DB
piece_db = PieceDB("pieces/piece_db.jsonl")

# Find string_length pieces
print("String length pieces in DB:")
print("="*80)
for piece in piece_db.pieces:
    if 'string' in piece.piece_id.lower() and 'length' in piece.piece_id.lower():
        print(f"\nPiece ID: {piece.piece_id}")
        print(f"Executor: {piece.executor}")
        print(f"Slots: {piece.in_spec.slots}")
        print(f"Requires: {piece.in_spec.requires}")

# Test with pipeline
print("\n" + "="*80)
print("Testing with pipeline...")
print("="*80)

pipeline = VerantyxV6Enhanced(piece_db_path="pieces/piece_db.jsonl")

# Test question
question = "What is the length of 'hello'?"
result = pipeline.solve(question)

print(f"\nQuestion: {question}")
print(f"Answer: {result.get('answer')}")
print(f"Status: {result.get('status')}")

# Check IR
ir = result.get('ir', {})
print(f"\nIR:")
print(f"  Domain: {ir.get('domain')}")
print(f"  Task: {ir.get('task')}")
print(f"  Entities: {ir.get('entities', [])}")
print(f"  source_text: {ir.get('metadata', {}).get('source_text', '')}")

# Check pieces found
pieces_found = result.get('pieces_found', [])
print(f"\nPieces Found: {len(pieces_found)}")
if pieces_found:
    for p in pieces_found[:3]:
        print(f"  - {p.get('piece_id')}: score={p.get('score', 0)}")

# Check execution
execution = result.get('execution', {})
print(f"\nExecution:")
print(f"  Executor: {execution.get('executor', 'N/A')}")
print(f"  Params: {execution.get('params', {})}")
print(f"  Result: {execution.get('result', 'N/A')}")
