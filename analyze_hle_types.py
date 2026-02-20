"""
HLE問題タイプ分析
"""

import json
import re
from collections import Counter

# HLEサンプルをロード
problems = []
with open("hle_100_sample.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            problems.append(json.loads(line))

print(f"Analyzing {len(problems)} problems\n")

# 問題タイプの分類
def classify_problem(text):
    """問題を分類"""
    text_lower = text.lower()
    
    # キーワードベースの分類
    if any(kw in text_lower for kw in ['compute', 'calculate', 'what is', 'find']):
        # 数値計算系
        if any(kw in text_lower for kw in ['bordism', 'cohomology', 'homology', 'homotopy']):
            return 'topology_advanced'
        elif any(kw in text_lower for kw in ['lie', 'algebra', 'group', 'representation']):
            return 'algebra_advanced'
        elif any(kw in text_lower for kw in ['curve', 'variety', 'scheme', 'sheaf']):
            return 'algebraic_geometry'
        elif any(kw in text_lower for kw in ['polynomial', 'root', 'equation']):
            return 'algebra_basic'
        elif re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', text):
            return 'arithmetic'
        else:
            return 'computation_other'
    
    elif any(kw in text_lower for kw in ['is it true', 'does', 'can', 'is']):
        # 判定系
        if any(kw in text_lower for kw in ['prime', 'divisible', 'gcd', 'lcm']):
            return 'number_theory'
        elif any(kw in text_lower for kw in ['tautology', 'valid', 'satisfiable']):
            return 'logic'
        else:
            return 'decision_other'
    
    elif any(kw in text_lower for kw in ['let', 'consider', 'suppose']):
        # 定義・設定系（通常は高度）
        return 'mathematical_definition'
    
    else:
        return 'other'

# 分類を実行
types = []
samples_by_type = {}

for i, problem in enumerate(problems):
    text = problem.get("problem_text", "")
    ptype = classify_problem(text)
    types.append(ptype)
    
    if ptype not in samples_by_type:
        samples_by_type[ptype] = []
    if len(samples_by_type[ptype]) < 2:  # 各タイプ2問まで
        samples_by_type[ptype].append((i, text[:100]))

# 統計
type_counts = Counter(types)

print("=" * 80)
print("Problem Type Distribution")
print("=" * 80)

for ptype, count in type_counts.most_common():
    pct = count / len(problems) * 100
    print(f"{ptype:30s}: {count:3d} ({pct:5.1f}%)")

print()
print("=" * 80)
print("Sample Problems by Type")
print("=" * 80)

for ptype, samples in sorted(samples_by_type.items()):
    print(f"\n{ptype}:")
    for idx, text in samples:
        print(f"  [{idx}] {text}...")

# 対応可能性評価
print("\n" + "=" * 80)
print("Feasibility Assessment")
print("=" * 80)

feasibility = {
    'arithmetic': ('HIGH', 'V6で対応可能'),
    'algebra_basic': ('MEDIUM', 'Executor追加で対応可能'),
    'number_theory': ('MEDIUM', 'Executor追加で対応可能'),
    'logic': ('HIGH', 'V6で対応済み'),
    'computation_other': ('LOW', '問題依存'),
    'decision_other': ('LOW', '問題依存'),
    'topology_advanced': ('VERY_LOW', '専門知識必要'),
    'algebra_advanced': ('VERY_LOW', '専門知識必要'),
    'algebraic_geometry': ('VERY_LOW', '専門知識必要'),
    'mathematical_definition': ('VERY_LOW', '高度な理論必要'),
    'other': ('UNKNOWN', '不明')
}

total_feasible = 0
for ptype, count in type_counts.items():
    level, note = feasibility.get(ptype, ('UNKNOWN', '不明'))
    if level in ['HIGH', 'MEDIUM']:
        total_feasible += count
    print(f"{ptype:30s}: {level:12s} - {note}")

print()
print(f"Potentially feasible problems: {total_feasible}/{len(problems)} ({total_feasible/len(problems)*100:.1f}%)")
