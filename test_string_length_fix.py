"""
Test string_length fix
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

# Initialize pipeline
pipeline = VerantyxV6Enhanced(piece_db_path="pieces/piece_db.jsonl")

# Test questions
test_questions = [
    ("What is the length of the string 'hello world'?", "11"),
    ("How many characters are in 'test'?", "4"),
    ("Count the characters in 'Python'", "6"),
]

print("Testing string_length fix...")
print("="*80)

correct = 0
for q, expected in test_questions:
    result = pipeline.solve(q)
    answer = result.get('answer')
    status = result.get('status')
    
    is_correct = str(answer) == expected
    correct += is_correct
    
    print(f"\nQ: {q}")
    print(f"Expected: {expected}")
    print(f"Answer: {answer}")
    print(f"Status: {status}")
    print(f"Result: {'✓' if is_correct else '✗'}")

print(f"\n{'='*80}")
print(f"Summary: {correct}/{len(test_questions)} correct")
print(f"Success: {correct == len(test_questions)}")
