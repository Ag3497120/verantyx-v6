"""
Executor Log Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import Piece, PieceInput, PieceOutput
from assembler.executor import Executor
import json

# Question
question = "Find the determinant of matrix [[2, 3], [1, 4]]."

# Decompose
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()

print("IR:")
print(json.dumps(ir_dict, indent=2))
print()

# Create piece manually
piece_data = {
    "piece_id": "linear_algebra_determinant",
    "executor": "executors.linear_algebra.matrix_determinant",
    "in_spec": {
        "requires": ["task:find", "domain:linear_algebra"],
        "slots": []
    },
    "out_spec": {
        "produces": ["decimal"],
        "schema": "decimal"
    },
    "tags": ["linear_algebra", "determinant", "matrix"]
}

in_spec = PieceInput(
    requires=piece_data["in_spec"]["requires"],
    slots=piece_data["in_spec"]["slots"]
)
out_spec = PieceOutput(
    produces=piece_data["out_spec"]["produces"],
    schema=piece_data["out_spec"]["schema"],
    artifacts=[]
)
piece = Piece(
    piece_id=piece_data["piece_id"],
    name="Matrix Determinant",
    description="行列式を計算",
    in_spec=in_spec,
    out_spec=out_spec,
    executor=piece_data["executor"],
    verifiers=[],
    cost={},
    confidence=1.0,
    tags=piece_data["tags"],
    examples=[]
)

print("Piece:", piece.piece_id)
print("Executor:", piece.executor)
print()

# Execute
executor = Executor()
pieces = [piece]

print("Executing...")
candidate = executor.execute_path(pieces, ir_dict)

print()
print("Result:")
if candidate:
    print(f"  Value: {candidate.fields.get('value')}")
    print(f"  Confidence: {candidate.confidence}")
    print(f"  Schema: {candidate.schema}")
else:
    print("  None (execution failed)")
