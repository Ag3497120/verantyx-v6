"""
Test multiple choice problem detection and solving
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from decomposer.decomposer import RuleBasedDecomposer

# Test question from HLE dataset
test_question = """Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?

Answer Choices:
A. Egalitarian Dominance
B. General Non-Extreme Priority
C. Non-Elitism
D. Weak Non-Sadism
E. Weak Quality Addition"""

expected_answer = "D"

print("=" * 80)
print("Multiple Choice Test")
print("=" * 80)

# Step 1: Test Decomposer
print("\nStep 1: Decomposer Test")
print("-" * 80)
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(test_question)
ir_dict = ir.to_dict()

print(f"Domain: {ir_dict['domain']}")
print(f"Task: {ir_dict['task']}")
print(f"Answer Schema: {ir_dict['answer_schema']}")
print(f"Options: {ir_dict.get('options', [])}")

# Check if domain is MULTIPLE_CHOICE
is_mc_domain = ir_dict['domain'] == 'multiple_choice'
is_option_label = ir_dict['answer_schema'] == 'option_label'
has_options = len(ir_dict.get('options', [])) > 0

print(f"\n✓ Domain == MULTIPLE_CHOICE: {is_mc_domain}")
print(f"✓ Answer Schema == option_label: {is_option_label}")
print(f"✓ Has Options: {has_options} ({len(ir_dict.get('options', []))} options)")

# Step 2: Test Pipeline
print("\nStep 2: Pipeline Test")
print("-" * 80)
pipeline = VerantyxV6Enhanced(piece_db_path="pieces/piece_db.jsonl")
result = pipeline.solve(test_question)

answer = result.get('answer')
status = result.get('status')

print(f"Answer: {answer}")
print(f"Expected: {expected_answer}")
print(f"Status: {status}")
print(f"Correct: {answer == expected_answer}")

# Step 3: Check what pieces were found
pieces_found = result.get('pieces_found', [])
print(f"\nPieces Found: {len(pieces_found)}")
if pieces_found:
    for i, p in enumerate(pieces_found[:5], 1):
        print(f"  {i}. {p.get('piece_id')} (score={p.get('score', 0):.2f})")

# Step 4: Check execution
execution = result.get('execution', {})
print(f"\nExecution:")
print(f"  Executor: {execution.get('executor', 'N/A')}")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
if answer == expected_answer:
    print("✓ SUCCESS: Multiple choice problem solved correctly")
else:
    print("✗ FAILED: Incorrect answer or no answer")
    print(f"  Expected: {expected_answer}")
    print(f"  Got: {answer}")
