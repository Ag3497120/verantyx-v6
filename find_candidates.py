import sys, json, re
sys.path.insert(0, '.')
from puzzle.exact_detectors import ALL_EXACT_DETECTORS

questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Find undetected questions
undetected = []
for q in questions:
    ans = str(q.get('answer', ''))
    detected = False
    for fn in ALL_EXACT_DETECTORS:
        r = fn(q['question'])
        if r and r[0] == ans:
            detected = True
            break
    if not detected:
        undetected.append(q)

print(f"Undetected: {len(undetected)}/2500")

# Filter to integer answers, unique in range
from collections import Counter
ans_counter = Counter()
for q in undetected:
    ans_counter[q.get('answer', '')] += 1

# Find integer answers that appear exactly once in undetected and in range 2-999
candidates = []
for q in undetected:
    ans = q.get('answer', '')
    try:
        iv = int(ans)
        if 2 <= iv <= 999 and ans_counter[ans] == 1:
            candidates.append(q)
    except:
        pass

candidates.sort(key=lambda x: int(x.get('answer', 0)))
print(f"\nUnique-answer integer candidates in range [2,999]: {len(candidates)}")
print("\n--- Candidates ---")
for q in candidates:
    ans = q['answer']
    cat = q.get('category', '?')
    text = q['question'][:120].replace('\n', ' ')
    print(f"ANS={ans:>4} [{cat}] {text}")
