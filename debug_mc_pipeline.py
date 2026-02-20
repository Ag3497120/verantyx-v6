"""
Debug multiple choice pipeline execution
"""
import sys
import json
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

test_question = """Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?

Answer Choices:
A. Egalitarian Dominance
B. General Non-Extreme Priority
C. Non-Elitism
D. Weak Non-Sadism
E. Weak Quality Addition"""

print("Debug: Multiple Choice Pipeline Execution")
print("=" * 80)

# Initialize pipeline
pipeline = VerantyxV6Enhanced(piece_db_path="pieces/piece_db.jsonl")

# Execute
result = pipeline.solve(test_question)

# Print full result
print("\nFull Result:")
print(json.dumps(result, indent=2, default=str))

# Check key fields
print("\n" + "=" * 80)
print("Key Fields:")
print("-" * 80)
print(f"status: {result.get('status')}")
print(f"answer: {result.get('answer')}")
print(f"method: {result.get('method')}")
print(f"confidence: {result.get('confidence')}")
print(f"error: {result.get('error')}")

# Check pieces_found in different locations
print("\n" + "=" * 80)
print("Pieces Information:")
print("-" * 80)

# 1. Top-level pieces_found
pieces_found = result.get('pieces_found', [])
print(f"result['pieces_found']: {len(pieces_found)} items")
if pieces_found:
    for p in pieces_found[:3]:
        print(f"  - {p}")

# 2. Evidence
evidence = result.get('evidence', [])
print(f"result['evidence']: {len(evidence)} items")
if evidence:
    for e in evidence[:3]:
        print(f"  - {e}")

# 3. Trace
trace = result.get('trace', [])
print(f"result['trace']: {len(trace)} items")
for t in trace:
    if 'piece' in t.lower():
        print(f"  - {t}")

# 4. IR
ir = result.get('ir', {})
print(f"\nresult['ir']:")
print(f"  domain: {ir.get('domain')}")
print(f"  answer_schema: {ir.get('answer_schema')}")
