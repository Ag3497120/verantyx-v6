"""Analyze per-problem results to find patterns"""
import json
from collections import Counter, defaultdict

# Load per-problem results
with open('hle_2500_per_problem.json') as f:
    results = json.load(f)

# Load original questions to get more context
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

print("="*80)
print("FAILURE ANALYSIS")
print("="*80)

# Overall stats
total = len(results)
correct = sum(1 for r in results if r['correct'])
incorrect = total - correct

print(f"\nTotal: {total}")
print(f"Correct: {correct} ({correct/total*100:.2f}%)")
print(f"Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")

# Method distribution for incorrect answers
print("\n" + "="*80)
print("METHODS FOR INCORRECT ANSWERS")
print("="*80)
incorrect_methods = Counter()
for r in results:
    if not r['correct']:
        method = r.get('method', 'unknown')
        incorrect_methods[method] += 1

for method, count in incorrect_methods.most_common(20):
    print(f"  {count:4d}  {method}")

# Category breakdown for incorrect
print("\n" + "="*80)
print("INCORRECT BY CATEGORY")
print("="*80)
category_incorrect = defaultdict(lambda: {'total': 0, 'incorrect': 0})
for r in results:
    cat = r['category']
    category_incorrect[cat]['total'] += 1
    if not r['correct']:
        category_incorrect[cat]['incorrect'] += 1

for cat, stats in sorted(category_incorrect.items(), key=lambda x: x[1]['incorrect'], reverse=True):
    pct = stats['incorrect']/stats['total']*100
    print(f"  {cat}: {stats['incorrect']}/{stats['total']} ({pct:.1f}%)")

# Find cases where answer was given but wrong (false positives)
print("\n" + "="*80)
print("FALSE POSITIVES (answered but wrong)")
print("="*80)
false_positives = []
for i, r in enumerate(results):
    if not r['correct'] and r['answer'] is not None and r['answer'] != '—':
        false_positives.append((i, r))

print(f"Found {len(false_positives)} false positives")
if false_positives:
    print("\nFirst 10 false positives:")
    for idx, (i, r) in enumerate(false_positives[:10]):
        q = questions[i]
        print(f"\n{idx+1}. Question {i} [{r['category']}] [{r['method']}]")
        print(f"   Q: {q['question'][:100]}...")
        print(f"   Expected: {r['expected']}")
        print(f"   Got: {r['answer']}")

# Find patterns in question text for failures
print("\n" + "="*80)
print("CHECKING FOR ALGEBRA EXECUTOR CALLS")
print("="*80)
algebra_failures = []
for i, r in enumerate(results):
    if not r['correct']:
        q = questions[i]
        q_text = q['question'].lower()
        # Look for equation-like patterns
        if any(pattern in q_text for pattern in ['solve', 'equation', '= 0', 'x =', 'factor', 'partition']):
            algebra_failures.append((i, r, q))

print(f"Found {len(algebra_failures)} potential algebra failures")
if algebra_failures:
    print("\nFirst 5 algebra-related failures:")
    for idx, (i, r, q) in enumerate(algebra_failures[:5]):
        print(f"\n{idx+1}. Question {i} [{r['category']}] [{r['method']}]")
        print(f"   Q: {q['question'][:120]}...")
        print(f"   Expected: {r['expected']}")
        print(f"   Got: {r['answer']}")

# Check for questions with numeric expected answers that we got wrong
print("\n" + "="*80)
print("NUMERIC ANSWER FAILURES (should be solvable)")
print("="*80)
numeric_failures = []
for i, r in enumerate(results):
    if not r['correct']:
        q = questions[i]
        expected = r['expected']
        # Check if expected is numeric
        try:
            float(expected)
            numeric_failures.append((i, r, q))
        except:
            pass

print(f"Found {len(numeric_failures)} failures with numeric expected answers")
print(f"These are {len(numeric_failures)/incorrect*100:.1f}% of all failures")

# Sample some
print("\nSample of 5 numeric failures:")
import random
random.seed(42)
sample = random.sample(numeric_failures, min(5, len(numeric_failures)))
for idx, (i, r, q) in enumerate(sample):
    print(f"\n{idx+1}. Question {i} [{r['category']}]")
    print(f"   Q: {q['question'][:120]}...")
    print(f"   Expected: {r['expected']}")
    print(f"   Method: {r['method']}")

print("\n" + "="*80)
