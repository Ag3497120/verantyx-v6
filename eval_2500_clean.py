"""
HLE 2500 評価 — カンニングなし版
PieceDB空 + 知識パイプライン（Qwen2.5）のみ
"""
import sys, os, json, time
from collections import Counter

os.environ['DISABLE_CONCEPT_BOOST'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

# 空の PieceDB で初期化
print("Loading HLE 2500...")
questions = []
with open("hle_2500_eval.jsonl", 'r') as f:
    for line in f:
        questions.append(json.loads(line))
print(f"Loaded {len(questions)} questions")

# 空 PieceDB
print("Initializing pipeline (empty PieceDB, knowledge pipeline only)...")
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db_empty.jsonl",
    use_llm_decomposer=False,
)
print("Ready\n")

correct = 0
total = 0
category_stats = {}
method_stats = Counter()
per_problem = []
t0 = time.time()

for i, q in enumerate(questions):
    category = q.get('category', 'Unknown')
    if category not in category_stats:
        category_stats[category] = {'total': 0, 'correct': 0}
    category_stats[category]['total'] += 1
    total += 1

    try:
        result = pipeline.solve(q['question'])
        answer = result.get('answer')
        expected = q['answer']
        method = result.get('method', 'unknown')

        is_correct = False
        if answer and answer not in (None, '—', 'None'):
            is_correct = flexible_match(str(answer), str(expected))

        if is_correct:
            correct += 1
            category_stats[category]['correct'] += 1

        method_stats[method] += 1
        per_problem.append({
            'idx': i,
            'category': category,
            'answer': str(answer) if answer else None,
            'expected': str(expected),
            'correct': is_correct,
            'method': method,
        })
    except Exception as e:
        per_problem.append({
            'idx': i,
            'category': category,
            'answer': None,
            'expected': q['answer'],
            'correct': False,
            'method': f'error:{e}',
        })

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta = rate * (len(questions) - i - 1)
        pct = correct / total * 100
        print(f"[{i+1}/{len(questions)}] {correct}/{total} ({pct:.1f}%)  "
              f"{elapsed:.0f}s elapsed  ETA {eta/60:.0f}min  ({rate:.1f}s/q)")

elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"FINAL: {correct}/{total} ({correct/total*100:.2f}%)")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Speed: {elapsed/total:.1f}s/question")
print(f"\nMethods: {dict(method_stats)}")
print(f"\nCategory breakdown:")
for cat, s in sorted(category_stats.items(), key=lambda x: -x[1]['total']):
    pct = s['correct']/s['total']*100 if s['total'] > 0 else 0
    print(f"  {cat}: {s['correct']}/{s['total']} ({pct:.1f}%)")

print(f"\nPipeline stats:")
for k, v in sorted(pipeline.stats.items()):
    if v and v != 0:
        print(f"  {k}: {v}")

# 結果保存
with open('hle_2500_clean_eval.json', 'w') as f:
    json.dump({
        'score': correct/total,
        'correct': correct,
        'total': total,
        'elapsed_s': elapsed,
        'per_problem': per_problem,
        'stats': {k: v for k, v in pipeline.stats.items() if v},
    }, f, indent=2, ensure_ascii=False)
print(f"\n✅ Results saved to hle_2500_clean_eval.json")
