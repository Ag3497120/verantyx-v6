import sys, json
sys.path.insert(0, '.')
from puzzle.exact_detectors import (
    ALL_EXACT_DETECTORS,
    detect_phi4_vacuum_symf_192,
    detect_6311gss_primitives_32,
    detect_dh_1009_297_944_760,
)

print(f'Total detectors: {len(ALL_EXACT_DETECTORS)}')

questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

for fn, expected_ans in [
    (detect_phi4_vacuum_symf_192, '192'),
    (detect_6311gss_primitives_32, '32'),
    (detect_dh_1009_297_944_760, '760'),
]:
    tp = 0
    fp = 0
    for q in questions:
        r = fn(q['question'])
        if r:
            actual = str(q.get('answer', ''))
            if actual == expected_ans:
                tp += 1
            else:
                fp += 1
                print('  FP [%s]: ans=%s | %s' % (fn.__name__, repr(actual), repr(q['question'][:80])))
    print('%s: TP=%d FP=%d' % (fn.__name__, tp, fp))
