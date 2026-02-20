#!/usr/bin/env python3
"""
Equation Solver用のピースを追加
"""
import json
from pathlib import Path

# 新しいピースの定義
new_pieces = [
    {
        "piece_id": "algebra_solve_linear",
        "name": "Linear Equation Solver",
        "description": "Solve linear equations like 2x + 3 = 11",
        "in": {
            "requires": ["domain:algebra", "task:find"],
            "slots": ["equation"],
            "optional": []
        },
        "out": {
            "produces": ["number"],
            "schema": "number",
            "artifacts": []
        },
        "executor": "executors.equation_solver.algebra_solve_equation",
        "verifiers": ["answer_schema"],
        "cost": {"time": "medium", "space": "low", "explosion_risk": "low"},
        "confidence": 0.9,
        "tags": ["algebra", "equation", "linear"],
        "examples": [{"question": "Solve 2x + 3 = 11", "answer": "4"}]
    },
    {
        "piece_id": "algebra_solve_equation_compute",
        "name": "Equation Solver (Compute)",
        "description": "Solve algebraic equations",
        "in": {
            "requires": ["domain:algebra", "task:compute"],
            "slots": [],
            "optional": []
        },
        "out": {
            "produces": ["number"],
            "schema": "number",
            "artifacts": []
        },
        "executor": "executors.equation_solver.algebra_solve_equation",
        "verifiers": [],
        "cost": {"time": "medium", "space": "low", "explosion_risk": "low"},
        "confidence": 0.85,
        "tags": ["algebra", "equation"],
        "examples": []
    },
    {
        "piece_id": "algebra_solve_quadratic",
        "name": "Quadratic Equation Solver",
        "description": "Solve quadratic equations ax² + bx + c = 0",
        "in": {
            "requires": ["domain:algebra"],
            "slots": ["a", "b", "c"],
            "optional": []
        },
        "out": {
            "produces": ["number_list"],
            "schema": "list",
            "artifacts": []
        },
        "executor": "executors.equation_solver.solve_quadratic_equation",
        "verifiers": [],
        "cost": {"time": "low", "space": "low", "explosion_risk": "low"},
        "confidence": 0.95,
        "tags": ["algebra", "quadratic"],
        "examples": []
    }
]

# piece_db.jsonlに追加
piece_db_path = Path(__file__).parent.parent / "pieces" / "piece_db.jsonl"

print("=" * 70)
print("Adding Equation Solver Pieces")
print("=" * 70)

# 既存のIDを確認（重複回避）
existing_ids = set()
if piece_db_path.exists():
    with open(piece_db_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                existing_ids.add(data.get('piece_id'))

# 新しいピースを追加
added = 0
with open(piece_db_path, 'a') as f:
    for piece in new_pieces:
        if piece['piece_id'] not in existing_ids:
            f.write(json.dumps(piece) + '\n')
            print(f"  ✓ Added: {piece['piece_id']}")
            added += 1
        else:
            print(f"  - Skipped (exists): {piece['piece_id']}")

print(f"\n✅ Added {added} new pieces")
print("=" * 70)
