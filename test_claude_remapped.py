#!/usr/bin/env python3
"""
Claude remapped公理のクイックテスト
"""
import json
from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB
from assembler.beam_search import GreedyAssembler
from assembler.executor import Executor
from grammar.composer import AnswerComposer

# 初期化
decomposer = RuleBasedDecomposer()
piece_db = PieceDB("pieces/piece_db.jsonl")
assembler = GreedyAssembler(piece_db)
executor = Executor()
composer = AnswerComposer()

print("=" * 70)
print("Claude Remapped Axiom Test")
print("=" * 70)
print(f"Loaded pieces: {len(piece_db.pieces)}")

# Claude公理の数
claude_pieces = [p for p in piece_db.pieces if any(x in p.piece_id for x in ['algebra:', 'calculus:', 'geometry:', 'chemistry:', 'physics:', 'probability:', 'statistics:'])]
print(f"Claude pieces: {len(claude_pieces)}")
print()

# テスト問題
test_cases = [
    {"question": "What is 2 + 2?", "expected": "4"},
    {"question": "Calculate 3 * 4", "expected": "12"},
    {"question": "What is 5 squared?", "expected": "25"},
    {"question": "What is 10 - 3?", "expected": "7"},
    {"question": "Calculate 8 / 2", "expected": "4"},
    {"question": "What is 2^3?", "expected": "8"},
    {"question": "Calculate factorial of 5", "expected": "120"},
    {"question": "What is the square root of 16?", "expected": "4"},
    {"question": "What is 15 mod 4?", "expected": "3"},
    {"question": "Calculate 7 + 8", "expected": "15"},
]

results = []
for i, test in enumerate(test_cases, 1):
    try:
        # Decompose
        ir = decomposer.decompose(test['question'])
        
        # Assemble
        pieces = assembler.assemble(ir)
        
        # Claude公理使用チェック
        claude_used = any(any(x in p.piece_id for x in ['algebra:', 'calculus:', 'geometry:']) for p in pieces)
        
        # Execute
        exec_results = []
        for piece in pieces:
            try:
                result = executor.execute(piece, ir.to_dict())
                if result and result.get('success'):
                    exec_results.append(result)
            except Exception as e:
                pass
        
        # Compose
        if exec_results:
            answer = composer.compose(exec_results, ir)
        else:
            answer = "Unable to solve"
        
        # Verify
        verified = str(answer).strip() == str(test['expected']).strip()
        
        result = {
            "question": test['question'],
            "expected": test['expected'],
            "answer": answer,
            "verified": verified,
            "pieces_used": len(pieces),
            "claude_used": claude_used
        }
        results.append(result)
        
        status = "✓" if verified else "✗"
        claude_mark = "🎓" if claude_used else "  "
        print(f"[{i:2d}] {status} {claude_mark} {test['question']}")
        print(f"     Expected: {test['expected']}, Got: {answer}")
        
    except Exception as e:
        print(f"[{i:2d}] ✗   {test['question']}")
        print(f"     ERROR: {str(e)[:60]}")
        results.append({
            "question": test['question'],
            "expected": test['expected'],
            "answer": f"ERROR: {str(e)}",
            "verified": False,
            "pieces_used": 0,
            "claude_used": False
        })

print()
print("=" * 70)
print("Summary")
print("=" * 70)

correct = sum(1 for r in results if r['verified'])
claude_used_count = sum(1 for r in results if r['claude_used'])

print(f"Correct: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")
print(f"Claude axioms used: {claude_used_count}/{len(results)}")
print()

# 保存
with open('test_claude_remapped_results.json', 'w') as f:
    json.dump({"results": results, "summary": {"correct": correct, "total": len(results), "claude_used": claude_used_count}}, f, indent=2)

print(f"Results saved to: test_claude_remapped_results.json")
print("=" * 70)
