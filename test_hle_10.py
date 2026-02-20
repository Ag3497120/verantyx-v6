"""
HLE 10問テスト
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced

# 問題をロード
problems = []
with open("hle_100_sample.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            problems.append(json.loads(line))

# 最初の10問のみ
problems = problems[:10]

print(f"Testing {len(problems)} problems\n")

v6 = VerantyxV6Enhanced()

verified = 0
for i, problem in enumerate(problems):
    question = problem.get("problem_text", "")
    
    print(f"\n[{i+1}/10] {question[:80]}...")
    
    try:
        # expected_answerなしで解く
        result = v6.solve(question, expected_answer=None, use_crystal=False)
        
        print(f"  Status: {result['status']}")
        print(f"  Answer: {result.get('answer', 'N/A')}")
        
        if result["status"] == "SOLVED":
            verified += 1
            
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*80}")
print(f"Results: {verified}/10 solved")
print(f"{'='*80}")
