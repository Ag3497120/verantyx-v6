#!/usr/bin/env python3
"""
Claude remapped公理の簡易テスト（ピース選択のみ）
"""
from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB

# 初期化
decomposer = RuleBasedDecomposer()
piece_db = PieceDB("pieces/piece_db.jsonl")

print("=" * 70)
print("Claude Remapped Axiom Test (Simplified)")
print("=" * 70)
print(f"Total pieces: {len(piece_db.pieces)}")

# Claude公理の数
claude_pieces = [p for p in piece_db.pieces if any(x in p.piece_id for x in ['algebra:', 'calculus:', 'geometry:', 'chemistry:', 'physics:', 'probability:', 'statistics:', 'number_theory:', 'combinatorics:'])]
print(f"Claude pieces: {len(claude_pieces)}")

# Executor分布
executors = {}
for p in piece_db.pieces:
    executors[p.executor] = executors.get(p.executor, 0) + 1

print(f"\nTop executors:")
for executor, count in sorted(executors.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {executor}: {count}")

print("\n" + "=" * 70)
print("Testing piece selection...")
print("=" * 70)

# テスト問題
test_cases = [
    "What is 2 + 2?",
    "Calculate the derivative of x^2",
    "Find the area of a circle with radius 5",
    "What is 5 factorial?",
    "Calculate the probability of getting heads",
]

for i, question in enumerate(test_cases, 1):
    print(f"\n[{i}] {question}")
    
    # Decompose
    ir = decomposer.decompose(question)
    print(f"  Domain: {ir.domain}, Task: {ir.task}")
    
    # Search pieces
    matches = piece_db.search(ir.to_dict(), top_k=5)
    
    print(f"  Top 3 pieces:")
    for piece, score in matches[:3]:
        claude_mark = "🎓" if any(x in piece.piece_id for x in ['algebra:', 'calculus:', 'geometry:']) else "  "
        print(f"    {claude_mark} {piece.piece_id[:40]:<40} (score: {score:.3f}, executor: {piece.executor})")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
