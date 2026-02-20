"""
V5が成功した問題を抽出
"""

import json
import sys

# V5の結果ファイルを読む
v5_results_path = "../verantyx_v5/v5_hle_2500_with_knowledge.jsonl"
hle_data_path = "../../avh_math/avh_math/db/hle_math_cross.jsonl"

# V5の成功問題IDを取得
v5_success_ids = set()
with open(v5_results_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            result = json.loads(line)
            if result.get("status") == "VERIFIED":
                problem_id = result.get("problem_id")
                if problem_id:
                    v5_success_ids.add(problem_id)

print(f"V5 VERIFIED problems: {len(v5_success_ids)}")

# HLEデータから該当問題を抽出
success_problems = []
with open(hle_data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if line.strip():
            problem = json.loads(line)
            problem_id = f"problem_{i}"
            
            if problem_id in v5_success_ids:
                success_problems.append({
                    "problem_id": problem_id,
                    "question": problem.get("problem_text", ""),
                    "index": i
                })

print(f"Extracted {len(success_problems)} problems")

# 保存
with open("v5_success_problems.jsonl", 'w', encoding='utf-8') as f:
    for problem in success_problems:
        f.write(json.dumps(problem, ensure_ascii=False) + '\n')

print("Saved to v5_success_problems.jsonl")

# 最初の5問を表示
print("\nFirst 5 problems:")
for problem in success_problems[:5]:
    print(f"{problem['problem_id']}: {problem['question'][:80]}...")
