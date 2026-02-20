import sys, json
sys.path.insert(0, '.')
from puzzle.exact_detectors import detect_ci22_cp102_cohomology_104

questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Test both known H^100 cohomology questions
tp_count = 0
fp_count = 0
for q in questions:
    r = detect_ci22_cp102_cohomology_104(q['question'])
    if r:
        ans = str(q.get('answer', ''))
        if ans == '104':
            tp_count += 1
            print(f"TP: {q['question'][:80]!r}")
        else:
            fp_count += 1
            print(f"FP: ans={ans!r} | {q['question'][:80]!r}")

print(f"\nTP={tp_count} FP={fp_count}")
