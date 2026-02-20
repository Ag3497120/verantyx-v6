"""Analyze CEGIS false positives - these are the biggest bug"""
import json

# Load per-problem results
with open('hle_2500_per_problem.json') as f:
    results = json.load(f)

# Load original questions
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

print("="*80)
print("CEGIS FALSE POSITIVES ANALYSIS")
print("="*80)

# Find all CEGIS false positives
cegis_wrong = []
cegis_correct = []

for i, r in enumerate(results):
    if r['method'] == 'cegis_proved':
        if r['correct']:
            cegis_correct.append((i, r, questions[i]))
        else:
            cegis_wrong.append((i, r, questions[i]))

print(f"\nCEGIS Stats:")
print(f"  Total CEGIS proved: {len(cegis_wrong) + len(cegis_correct)}")
print(f"  Correct: {len(cegis_correct)}")
print(f"  WRONG: {len(cegis_wrong)}")
print(f"  Accuracy: {len(cegis_correct)/(len(cegis_wrong)+len(cegis_correct))*100:.1f}%")

# This is terrible! CEGIS is only 11% accurate!
# 69 correct, 541 wrong = 11% accuracy

print("\n" + "="*80)
print("SAMPLE OF CEGIS WRONG ANSWERS")
print("="*80)

import random
random.seed(42)
sample = random.sample(cegis_wrong, min(20, len(cegis_wrong)))

for idx, (i, r, q) in enumerate(sample):
    print(f"\n{idx+1}. Question {i} [{r['category']}]")
    print(f"   Q: {q['question'][:150]}...")
    print(f"   Expected: {r['expected']}")
    print(f"   CEGIS gave: {r['answer']}")

    # Try to identify patterns
    expected_str = str(r['expected']).lower()
    answer_str = str(r['answer']).lower()

    # Check if it's a multiple choice question
    is_mcq = any(opt in expected_str for opt in ['a', 'b', 'c', 'd', 'e', 'f'])
    if is_mcq:
        print(f"   [MCQ detected]")

print("\n" + "="*80)
print("PATTERN: Are CEGIS errors mostly MCQ?")
print("="*80)

mcq_wrong = 0
non_mcq_wrong = 0

for i, r, q in cegis_wrong:
    expected = str(r['expected'])
    # Heuristic: if expected answer is a single letter A-Z, likely MCQ
    if len(expected) <= 2 and expected.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        mcq_wrong += 1
    else:
        non_mcq_wrong += 1

print(f"MCQ wrong: {mcq_wrong}")
print(f"Non-MCQ wrong: {non_mcq_wrong}")
print(f"MCQ ratio: {mcq_wrong/len(cegis_wrong)*100:.1f}%")

print("\n" + "="*80)
