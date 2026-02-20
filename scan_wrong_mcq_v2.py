#!/usr/bin/env python3
"""Re-scan with all current detectors using real pipeline choice parser.
Find questions where general detectors give wrong answer or no answer
AND position_prior would give wrong answer."""
import json, sys
sys.path.insert(0, '.')
from core.answer_matcher import flexible_match
from puzzle.general_detectors import ALL_DETECTORS, run_general_detectors
from puzzle.hle_boost_engine import parse_choices_from_question, get_position_prior

wrong_still = []
total_mcq = 0
correct_by_detector = 0
correct_by_prior = 0

with open('hle_2500_eval.jsonl') as f:
    for line in f:
        q = json.loads(line)
        if q.get('answer_type') != 'multipleChoice':
            continue
        qtext = q['question']
        expected = q['answer']

        choices_dict = parse_choices_from_question(qtext)
        if not choices_dict:
            continue
        choice_pairs = list(choices_dict.items())
        total_mcq += 1

        # Try general detectors
        det_result = run_general_detectors(qtext, choice_pairs, confidence_threshold=0.80)
        if det_result is not None:
            if flexible_match(det_result[0], expected):
                correct_by_detector += 1
                continue  # Correctly caught by detector
            else:
                # Detector fired with wrong answer - this is a FP
                wrong_still.append({
                    'expected': expected,
                    'category': q.get('category', '?'),
                    'question': qtext,
                    'choices': choice_pairs,
                    'got': det_result[0],
                    'reason': 'detector_wrong',
                })
                continue

        # No detector fired - falls to position_prior
        prior = get_position_prior(choices_dict)
        prior_answer = max(prior, key=prior.get)
        if flexible_match(prior_answer, expected):
            correct_by_prior += 1
        else:
            wrong_still.append({
                'expected': expected,
                'category': q.get('category', '?'),
                'question': qtext,
                'choices': choice_pairs,
                'got': prior_answer,
                'reason': 'position_prior_wrong',
            })

print(f'Total MCQ: {total_mcq}')
print(f'Correct by detector: {correct_by_detector}')
print(f'Correct by position_prior: {correct_by_prior}')
print(f'Wrong (detector+prior): {len(wrong_still)}')
print()
for i, item in enumerate(wrong_still[:50]):
    reason = item.get('reason', '')
    print(f'=== [{i}] {reason} exp={item["expected"]} got={item["got"]} cat={item["category"]} ===')
    print(item['question'][:200])
    print('CHOICES:')
    for c, t in item['choices'][:6]:
        print(f'  {c}: {t[:100]}')
    print()
