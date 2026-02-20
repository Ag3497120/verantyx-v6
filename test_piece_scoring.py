"""
Piece Scoring Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import Piece
import json

question = "Find the determinant of matrix [[2, 3], [1, 4]]."

decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()

print("IR:")
print(f"  Domain: {ir_dict['domain']}")
print(f"  Task: {ir_dict['task']}")
print(f"  Answer Schema: {ir_dict['answer_schema']}")
print(f"  Keywords: {ir_dict['metadata']['keywords']}")
print()

# Load pieces
pieces = []
with open("pieces/piece_db.jsonl", "r") as f:
    for line in f:
        piece_data = json.loads(line)
        pieces.append(piece_data)

# Focus on these pieces
focus_ids = ["linear_algebra_determinant", "nt_divisor_count_find"]

for piece_data in pieces:
    if piece_data["piece_id"] in focus_ids:
        # Create piece object
        from pieces.piece import PieceInput, PieceOutput
        in_spec = PieceInput(
            requires=piece_data["in"]["requires"],
            slots=piece_data["in"]["slots"]
        )
        out_spec = PieceOutput(
            produces=piece_data["out"]["produces"],
            schema=piece_data["out"]["schema"],
            artifacts=piece_data["out"].get("artifacts", [])
        )
        piece = Piece(
            piece_id=piece_data["piece_id"],
            name=piece_data["name"],
            description=piece_data["description"],
            in_spec=in_spec,
            out_spec=out_spec,
            executor=piece_data["executor"],
            verifiers=piece_data.get("verifiers", []),
            cost=piece_data.get("cost", {}),
            confidence=piece_data.get("confidence", 1.0),
            tags=piece_data.get("tags", []),
            examples=piece_data.get("examples", [])
        )
        
        score = piece.matches_ir(ir_dict)
        print(f"Piece: {piece.piece_id}")
        print(f"  Requires: {piece.in_spec.requires}")
        print(f"  Out schema: {piece.out_spec.schema}")
        print(f"  Tags: {piece.tags}")
        print(f"  Score: {score:.3f}")
        print()
