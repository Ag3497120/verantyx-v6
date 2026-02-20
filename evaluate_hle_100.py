#!/usr/bin/env python3
"""
HLE 100問クイックテスト（Phase 5G評価）
"""
import json
from pathlib import Path
from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB
from assembler.beam_search import GreedyAssembler
from assembler.executor import Executor
from grammar.composer import AnswerComposer, GrammarDB
from core.answer_matcher import flexible_match

print("=" * 70)
print("HLE 100-Question Quick Test (Phase 5G)")
print("=" * 70)

# データセット読み込み
dataset_path = Path("hle_2500_eval.jsonl")
if not dataset_path.exists():
    print("ERROR: hle_2500_eval.jsonl not found")
    exit(1)

questions = []
with open(dataset_path) as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

print(f"Loaded {len(questions)} questions")
print(f"Testing first 100 questions...")
print()

# パイプライン初期化
decomposer = RuleBasedDecomposer()
piece_db = PieceDB("pieces/piece_db.jsonl")
assembler = GreedyAssembler(piece_db)
executor = Executor()
grammar_db = GrammarDB("grammar/grammar_db.jsonl")
composer = AnswerComposer(grammar_db)

print(f"Pipeline initialized:")
print(f"  Pieces: {len(piece_db.pieces)}")
print()

# 評価
results = []
correct = 0
failed = 0

for i, q in enumerate(questions[:100], 1):
    try:
        # Decompose
        ir = decomposer.decompose(q['question'])
        
        # Assemble
        pieces = assembler.assemble(ir.to_dict(), ir.answer_schema)
        
        # Execute
        exec_results = []
        if pieces:
            for piece in pieces:
                try:
                    result = executor.execute(piece, ir.to_dict())
                    if result and result.get('success'):
                        exec_results.append(result)
                except:
                    pass
        
        # Compose
        if exec_results:
            answer = composer.compose(exec_results, ir)
        else:
            answer = "Unable to solve"
        
        # Verify（flexible_matchを使用）
        expected = q.get('answer', '')
        verified = flexible_match(answer, expected)
        
        if verified:
            correct += 1
        
        results.append({
            "question": q['question'],
            "expected": expected,
            "answer": answer,
            "verified": verified,
            "category": q.get('category', 'Unknown')
        })
        
        if i % 10 == 0:
            print(f"  Progress: {i}/100 (Correct: {correct}, {100*correct/i:.1f}%)")
    
    except Exception as e:
        failed += 1
        results.append({
            "question": q['question'],
            "expected": q.get('answer', ''),
            "answer": f"ERROR: {str(e)[:50]}",
            "verified": False,
            "category": q.get('category', 'Unknown')
        })

print()
print("=" * 70)
print("Results")
print("=" * 70)
print(f"Correct: {correct}/100 ({100*correct/100:.1f}%)")
print(f"Failed: {failed}/100")
print()

# カテゴリ別の結果
from collections import Counter
category_correct = Counter()
category_total = Counter()

for r in results:
    cat = r['category']
    category_total[cat] += 1
    if r['verified']:
        category_correct[cat] += 1

print("Category Breakdown:")
for cat in sorted(category_total.keys()):
    total = category_total[cat]
    correct_cat = category_correct[cat]
    print(f"  {cat}: {correct_cat}/{total} ({100*correct_cat/total:.1f}%)")

# 保存
output_path = Path("hle_100_phase5g_results.json")
with open(output_path, 'w') as f:
    json.dump({
        "results": results,
        "summary": {
            "total": 100,
            "correct": correct,
            "failed": failed,
            "accuracy": correct / 100
        }
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")
print("=" * 70)
