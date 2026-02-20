import sys, json
sys.path.insert(0, '.')
from puzzle.exact_detectors import ALL_EXACT_DETECTORS
from collections import Counter

questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Target answers to show full text for
targets = {
    '32', '192', '321', '84', '510', '273', '130', '97',
    '108', '106', '300', '385', '447', '760', '98', '350', '155'
}

for q in questions:
    ans = str(q.get('answer', ''))
    if ans in targets:
        # Check not detected
        detected = any(fn(q['question']) is not None for fn in ALL_EXACT_DETECTORS)
        if not detected:
            cat = q.get('category', '?')
            print(f"\n{'='*80}")
            print(f"ANS={ans} [{cat}]")
            print(q['question'][:600])
            print('...' if len(q['question']) > 600 else '')
