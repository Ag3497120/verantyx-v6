"""
HLEから対応可能な問題のサブセットを作成
"""

import json
import re

# HLE全体から対応可能な問題を抽出
def is_feasible(text):
    """V6で対応可能かチェック"""
    text_lower = text.lower()
    
    # HIGH feasibility
    # 基本的な算術
    if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', text):
        if not any(kw in text_lower for kw in ['bordism', 'cohomology', 'lie', 'scheme', 'variety']):
            return True
    
    # 論理問題
    if any(kw in text_lower for kw in ['tautology', 'valid', 'satisfiable', '->', 'implies']):
        if not any(kw in text_lower for kw in ['functional', 'measure', 'topology']):
            return True
    
    # MEDIUM feasibility
    # 基本的な数論
    if any(kw in text_lower for kw in ['prime', 'divisible', 'gcd']) and len(text) < 200:
        return True
    
    # 基本的な組み合わせ
    if any(kw in text_lower for kw in ['permutation', 'combination', 'choose']):
        if not any(kw in text_lower for kw in ['moduli', 'algebraic', 'cohomology']):
            return True
    
    # 確率（基本的なもの）
    if 'probability' in text_lower and len(text) < 300:
        if not any(kw in text_lower for kw in ['functional', 'measure theory', 'stochastic']):
            return True
    
    return False

# サンプルから抽出
problems = []
with open("hle_100_sample.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            problems.append(json.loads(line))

feasible_problems = []
for i, problem in enumerate(problems):
    text = problem.get("problem_text", "")
    if is_feasible(text):
        feasible_problems.append({
            "index": i,
            "problem_text": text
        })

print(f"Feasible problems: {len(feasible_problems)}/{len(problems)}")
print()

# サンプルを表示
for i, prob in enumerate(feasible_problems[:20], 1):
    print(f"[{i}] (#{prob['index']}) {prob['problem_text'][:100]}...")

# 保存
with open("hle_feasible_subset.jsonl", 'w', encoding='utf-8') as f:
    for prob in feasible_problems:
        f.write(json.dumps(prob, ensure_ascii=False) + '\n')

print(f"\nSaved {len(feasible_problems)} problems to hle_feasible_subset.jsonl")
