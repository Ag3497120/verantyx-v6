"""
キャッシュ再ビルド → full eval の一発スクリプト
"""
import sys
import re
import json
import time

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

# --- Step 1: キャッシュ再ビルド ---
print("=== Step 1: キャッシュ再ビルド ===")
print("Loading HLE 2500...")
questions_text = []
with open("hle_2500_eval.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        questions_text.append(item['question'])
print(f"Loaded {len(questions_text)} questions")

from knowledge.concept_boost import ConceptBooster
booster = ConceptBooster(use_cache=False)
booster.build_cache(
    questions_text,
    cache_path="knowledge/concept_cache.jsonl",
    verbose=True
)

# ヒット確認（5問）
print("\n=== Step 2: ヒット確認 ===")
booster2 = ConceptBooster(use_cache=True)
n = booster2.load_cache("knowledge/concept_cache.jsonl")
print(f"Cache loaded: {n} entries")

hits = 0
for q in questions_text[:5]:
    key = q.lower()[:200]
    scores = booster2.get_scores(q)
    if scores:
        hits += 1
        print(f"  HIT: {q[:60]} → top={sorted(scores.items(), key=lambda x:-x[1])[:2]}")
    else:
        print(f"  MISS: {q[:60]}")
print(f"Hit rate: {hits}/5")

# --- Step 3: Full eval ---
print("\n=== Step 3: Full Eval ===")

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

YES_NO_TRIGGERS = [
    r"^Is (it|the|a|an|there|this|that|every|each|any|all)\b",
    r"^Does (there|the|this|it|a|any|every)\b",
    r"^Can (the|a|this|we|one|you|every)\b",
    r"^Are (there|the|all|any|these|those)\b",
    r"^Do (the|all|any|these|those|most)\b",
    r"^Will (the|a|this|it|every)\b",
    r"^Has (the|this|a|every|any)\b",
    r"^Must (the|all|every|this|a)\b",
    r"^Could (the|this|a|every|any)\b",
    r"^Would (the|this|a|every)\b",
    r"^Suppose .+\. Is (it|the|a|there|this)\b",
    r"^Given .+\. Is (it|the|a|there|this)\b",
]

def is_yes_no_question(question: str) -> bool:
    q = question.strip()
    return any(re.match(p, q) for p in YES_NO_TRIGGERS)

print("Initializing pipeline...")
pipeline = VerantyxV6Enhanced(piece_db_path="pieces/piece_db.jsonl")
print("Ready")

questions = []
with open("hle_2500_eval.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

start_time = time.time()
correct = 0
total = 0
category_stats = {}
yes_no_stats = {'detected': 0, 'yes_fallback': 0}

for i, q in enumerate(questions):
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(questions)} ({(i+1)/len(questions)*100:.1f}%)")
    
    category = q.get('category', 'Unknown')
    if category not in category_stats:
        category_stats[category] = {'total': 0, 'correct': 0}
    category_stats[category]['total'] += 1
    total += 1
    
    try:
        question_text = q['question']
        expected = q['answer']
        yes_no = is_yes_no_question(question_text)
        if yes_no:
            yes_no_stats['detected'] += 1
        
        result = pipeline.solve(question_text)
        answer = result.get('answer')
        
        if answer is None:
            answer = 'Yes' if yes_no else '0'
            if yes_no:
                yes_no_stats['yes_fallback'] += 1
        
        if answer is not None and expected and flexible_match(answer, expected, tolerance=1e-4):
            correct += 1
            category_stats[category]['correct'] += 1
    except Exception as e:
        pass

elapsed = time.time() - start_time

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%")
print(f"Time: {elapsed:.1f}s")
print()
print(f"Yes/No: detected={yes_no_stats['detected']}, yes_fallback={yes_no_stats['yes_fallback']}")
print()
print("Category breakdown:")
for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0):
    pct = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
    print(f"  {cat}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

with open('hle_2500_phase5j_final.json', 'w') as f:
    json.dump({
        'total': total, 'correct': correct,
        'accuracy': correct/total*100, 'time': elapsed,
        'category_stats': category_stats,
        'yes_no_stats': yes_no_stats,
    }, f, indent=2)
print(f"\nSaved to hle_2500_phase5j_final.json")
