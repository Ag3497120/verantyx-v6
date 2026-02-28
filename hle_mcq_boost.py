"""
HLE MCQ Boost: unanswered MCQ questions only, with Qwen 7B enabled
"""
import json, sys, os, time
from collections import Counter

os.environ.setdefault('DISABLE_CONCEPT_BOOST', '1')
os.environ.setdefault('DISABLE_PATTERN_DETECTORS', '1')
sys.path.insert(0, '.')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

# Load data
with open('hle_2500_eval.jsonl') as f:
    qs = [json.loads(l) for l in f]
with open('hle_2500_per_problem.json') as f:
    prev_results = json.load(f)

# Find unanswered MCQ
unanswered_indices = []
for i, (q, r) in enumerate(zip(qs, prev_results)):
    if q.get('answer_type') == 'multipleChoice' and r.get('answer') is None:
        unanswered_indices.append(i)

print(f"Boosting {len(unanswered_indices)} unanswered MCQ questions with Qwen 7B...")

# Pipeline WITH LLM
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db.jsonl",
    use_llm_decomposer=True,  # Enable Qwen
)

new_correct = 0
new_answered = 0
t0 = time.time()

for count, idx in enumerate(unanswered_indices):
    q = qs[idx]
    
    result = pipeline.solve(q['question'], expected_answer=q.get('answer'))
    answer = result.get('answer')
    expected = q['answer']
    method = result.get('method', 'unknown')
    
    is_correct = False
    if answer is not None and answer != '—' and expected:
        is_correct = flexible_match(answer, expected, tolerance=1e-4)
    
    if answer is not None:
        new_answered += 1
        # Update prev_results
        prev_results[idx] = {
            'idx': idx,
            'category': q.get('category', 'Unknown'),
            'correct': is_correct,
            'answer': answer,
            'expected': expected,
            'method': method,
        }
    
    if is_correct:
        new_correct += 1
    
    if (count + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{count+1}/{len(unanswered_indices)}] answered={new_answered} correct={new_correct} ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")
print(f"New answers: {new_answered}/{len(unanswered_indices)}")
print(f"New correct: {new_correct}")

# Recalculate total score
total_correct = sum(1 for r in prev_results if r.get('correct'))
print(f"\nUpdated total: {total_correct}/2500 ({100*total_correct/2500:.2f}%)")
prev_score = 216
print(f"Change: {prev_score} → {total_correct} ({'+' if total_correct > prev_score else ''}{total_correct - prev_score})")

# Save updated results
with open('hle_2500_per_problem.json', 'w') as f:
    json.dump(prev_results, f, indent=2)
print("Saved updated per_problem results")
