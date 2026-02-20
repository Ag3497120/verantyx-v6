"""Deep dive into a few CEGIS failures to understand the bug"""
import json
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

# Load per-problem results
with open('hle_2500_per_problem.json') as f:
    results = json.load(f)

# Load questions
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Find some CEGIS failures
cegis_failures = []
for i, r in enumerate(results):
    if r['method'] == 'cegis_proved' and not r['correct']:
        cegis_failures.append((i, r, questions[i]))

print("="*80)
print("DEEP DIVE: Re-run a few CEGIS failures with debug output")
print("="*80)

# Initialize pipeline with debug
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db.jsonl",
    use_llm_decomposer=False,
)

# Test a few failures
import random
random.seed(42)
sample_failures = random.sample(cegis_failures, min(3, len(cegis_failures)))

for idx, (i, r, q) in enumerate(sample_failures):
    print(f"\n{'='*80}")
    print(f"CASE {idx+1}: Question {i} [{r['category']}]")
    print(f"{'='*80}")
    print(f"Question: {q['question'][:200]}...")
    print(f"Expected: {r['expected']}")
    print(f"Previous CEGIS answer: {r['answer']}")
    print()

    # Re-run with debug
    try:
        result = pipeline.solve(q['question'])
        print(f"Re-run answer: {result.get('answer')}")
        print(f"Re-run method: {result.get('method')}")
        print(f"Re-run confidence: {result.get('confidence')}")

        # Check if CEGIS info is available
        if 'debug' in result or 'cegis_info' in result:
            print(f"Debug info: {result.get('debug', result.get('cegis_info', {}))}")
    except Exception as e:
        print(f"ERROR re-running: {e}")

    print()

print("="*80)
print("ANALYSIS: Check certificate checker logic")
print("="*80)

# Let's check the certificate checker
from cegis.certificate import CertificateChecker, Certificate, CertKind

checker = CertificateChecker()

# Create a fake certificate with HIGH_CONFIDENCE
fake_cert_high = Certificate(
    kind=CertKind.HIGH_CONFIDENCE,
    value={"worlds_tested": 0, "passed": 0},
    confidence=0.8,
    details={"candidate": "some_wrong_answer"},
)

# Create a fake certificate with SMALL_WORLD
fake_cert_small = Certificate(
    kind=CertKind.SMALL_WORLD,
    value={"worlds_tested": 5, "passed": 5, "ratio": 1.0},
    confidence=0.9,
    details={"candidate": "some_answer"},
)

print(f"\nChecking HIGH_CONFIDENCE cert with worlds=0:")
result_high = checker.check(fake_cert_high, "some_wrong_answer")
print(f"  Passed: {result_high}")

print(f"\nChecking SMALL_WORLD cert with 5/5 worlds:")
result_small = checker.check(fake_cert_small, "some_answer")
print(f"  Passed: {result_small}")

print("\n" + "="*80)
