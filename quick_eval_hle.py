"""
Quick HLE 2500 evaluation
⚠️ バイアスなし版: 推論できない問題には回答しない
"""
import sys
import os
import json
import time
from collections import Counter

# 速度優先: concept_boost は事前キャッシュ依存だが cache miss が多い場合は無効化
# (live SVD search は ~0.07s/問 = 2500問で約175s の遅延)
os.environ.setdefault('DISABLE_CONCEPT_BOOST', '1')

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

# Load dataset
print("Loading HLE 2500...")
questions = []
with open("hle_2500_eval.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))
print(f"Loaded {len(questions)} questions")

# Initialize pipeline
print("Initializing pipeline...")
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db.jsonl",
    use_llm_decomposer=False,   # バルク評価時はOllamaをスキップ（速度優先）
)
print("Ready")

# Evaluate
print("\nEvaluating...")
start_time = time.time()
correct = 0
total = 0
category_stats = {}
method_stats = Counter()
per_problem = []  # per-problem tracking for debugging

for i, q in enumerate(questions):
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(questions)} ({(i+1)/len(questions)*100:.1f}%)")

    category = q.get('category', 'Unknown')
    if category not in category_stats:
        category_stats[category] = {'total': 0, 'correct': 0, 'methods': Counter()}

    category_stats[category]['total'] += 1
    total += 1

    try:
        result = pipeline.solve(q['question'])
        answer = result.get('answer')
        expected = q['answer']
        method = result.get('method', 'unknown')

        # ⚠️ バイアスなし: answer=None のときは何も答えない（'0'フォールバック禁止）
        # None/'—' は不正解扱い（柔軟マッチにかけない）
        is_correct = False
        if answer is not None and answer != '—' and expected:
            is_correct = flexible_match(answer, expected, tolerance=1e-4)

        if is_correct:
            correct += 1
            category_stats[category]['correct'] += 1
            category_stats[category]['methods'][method] += 1
            method_stats[method] += 1

        per_problem.append({
            'idx': i,
            'category': category,
            'correct': is_correct,
            'answer': str(answer) if answer is not None else None,
            'expected': expected,
            'method': method,
        })

    except Exception as e:
        per_problem.append({
            'idx': i,
            'category': category,
            'correct': False,
            'answer': None,
            'expected': q.get('answer'),
            'method': f'error:{e}',
        })

elapsed = time.time() - start_time

# Results
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%")
print(f"Time: {elapsed:.1f}s")
print()
print("Category breakdown:")
for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0):
    pct = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
    top_methods = stats['methods'].most_common(3)
    method_str = ', '.join(f"{m}×{c}" for m, c in top_methods)
    print(f"  {cat}: {stats['correct']}/{stats['total']} ({pct:.1f}%)  [{method_str}]")

print()
print("Top methods (correct answers):")
for method, cnt in method_stats.most_common(15):
    print(f"  {cnt:4d}  {method}")

# Save summary
with open('hle_2500_phase5h_final.json', 'w') as f:
    json.dump({
        'total': total,
        'correct': correct,
        'accuracy': correct/total*100,
        'time': elapsed,
        'category_stats': {k: {'total': v['total'], 'correct': v['correct']} for k, v in category_stats.items()},
        'method_stats': dict(method_stats.most_common(30)),
    }, f, indent=2)

# Save per-problem results for debugging
with open('hle_2500_per_problem.json', 'w') as f:
    json.dump(per_problem, f, indent=2, ensure_ascii=False)

print(f"\nSaved to hle_2500_phase5h_final.json")
print(f"Per-problem saved to hle_2500_per_problem.json")

# ── Auto-commit if score improved ─────────────────────────────────────────────
import subprocess

BEST_SCORE_FILE = 'best_score.json'
new_pct = correct / total * 100

# Load previous best
prev_best = 0.0
if os.path.exists(BEST_SCORE_FILE):
    try:
        with open(BEST_SCORE_FILE) as f:
            prev_best = json.load(f).get('accuracy', 0.0)
    except Exception:
        pass

if new_pct > prev_best:
    # Update best score record
    with open(BEST_SCORE_FILE, 'w') as f:
        json.dump({'accuracy': new_pct, 'correct': correct, 'total': total,
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S+09:00')}, f, indent=2)

    # Build commit message
    top_m = method_stats.most_common(3)
    method_summary = ', '.join(f"{m}×{c}" for m, c in top_m)
    commit_msg = (
        f"score: {new_pct:.2f}% bias-free ({correct}/{total}) "
        f"[+{new_pct - prev_best:.2f}pp vs prev {prev_best:.2f}%] "
        f"— {method_summary}"
    )

    print(f"\n🎉 スコア改善! {prev_best:.2f}% → {new_pct:.2f}% (+{new_pct-prev_best:.2f}pp)")
    print(f"📦 コミット中: {commit_msg}")

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        ['bash', 'commit_score.sh', f'{new_pct:.2f}', commit_msg],
        cwd=repo_dir, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️ push失敗: {result.stderr[:200]}")
else:
    print(f"\nスコア変化なし or 後退 ({prev_best:.2f}% → {new_pct:.2f}%)。コミットをスキップ。")
# ─────────────────────────────────────────────────────────────────────────────
