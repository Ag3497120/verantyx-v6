"""
HLE 2500問の完全分析
"""

import json
import re
from collections import Counter, defaultdict

# HLE全問題をロード
hle_path = "/Users/motonishikoudai/avh_math/avh_math/db/hle_math_cross.jsonl"

print("Loading HLE 2500 problems...")

problems = []
with open(hle_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if line.strip():
            data = json.loads(line)
            problems.append({
                "index": i,
                "text": data.get("problem_text", ""),
                "core": data.get("core", {})
            })
            if len(problems) >= 2500:
                break

print(f"Loaded {len(problems)} problems\n")

# 問題タイプの詳細分類
def classify_detailed(text, core):
    """詳細な分類"""
    text_lower = text.lower()
    concepts = core.get("concepts", [])
    numbers = core.get("numbers", [])
    
    # ドメイン分類
    domain = "unknown"
    
    # 算術・基本計算
    if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', text) and len(text) < 200:
        domain = "arithmetic_basic"
    
    # 数論
    elif any(kw in text_lower for kw in ['prime', 'divisor', 'gcd', 'lcm', 'congruent', 'modulo']):
        if any(kw in text_lower for kw in ['elliptic', 'algebraic number']):
            domain = "number_theory_advanced"
        else:
            domain = "number_theory_basic"
    
    # 組み合わせ論
    elif any(kw in text_lower for kw in ['permutation', 'combination', 'binomial', 'choose', 'arrange']):
        domain = "combinatorics"
    
    # 確率・統計
    elif any(kw in text_lower for kw in ['probability', 'random', 'expected', 'distribution', 'variance']):
        if any(kw in text_lower for kw in ['measure', 'stochastic process', 'martingale']):
            domain = "probability_advanced"
        else:
            domain = "probability_basic"
    
    # 代数
    elif any(kw in text_lower for kw in ['polynomial', 'equation', 'root', 'factor']):
        if any(kw in text_lower for kw in ['lie', 'representation', 'cohomology', 'homology']):
            domain = "algebra_advanced"
        else:
            domain = "algebra_basic"
    
    # 幾何
    elif any(kw in text_lower for kw in ['triangle', 'circle', 'square', 'polygon', 'angle', 'area', 'volume']):
        if any(kw in text_lower for kw in ['manifold', 'variety', 'scheme', 'sheaf']):
            domain = "geometry_advanced"
        else:
            domain = "geometry_basic"
    
    # 位相幾何学
    elif any(kw in text_lower for kw in ['bordism', 'cohomology', 'homology', 'homotopy', 'manifold']):
        domain = "topology"
    
    # 論理
    elif any(kw in text_lower for kw in ['tautology', 'valid', 'satisfiable', 'implies', '->']):
        domain = "logic"
    
    # リー代数・群論
    elif any(kw in text_lower for kw in ['lie algebra', 'lie group', 'representation', 'character']):
        domain = "lie_theory"
    
    # 代数幾何
    elif any(kw in text_lower for kw in ['variety', 'scheme', 'sheaf', 'moduli']):
        domain = "algebraic_geometry"
    
    # 関数解析
    elif any(kw in text_lower for kw in ['functional', 'operator', 'hilbert', 'banach', 'sobolev']):
        domain = "functional_analysis"
    
    # グラフ理論
    elif any(kw in text_lower for kw in ['graph', 'vertex', 'edge', 'tree', 'path', 'cycle']):
        domain = "graph_theory"
    
    # その他
    else:
        if 'integral' in text_lower or 'derivative' in text_lower:
            domain = "calculus"
        elif 'matrix' in text_lower or 'linear' in text_lower:
            domain = "linear_algebra"
        else:
            domain = "other"
    
    return domain

# 全問題を分類
print("Classifying problems...")
domains = []
domain_samples = defaultdict(list)

for prob in problems:
    domain = classify_detailed(prob["text"], prob["core"])
    domains.append(domain)
    
    # サンプル保存（各ドメイン3問まで）
    if len(domain_samples[domain]) < 3:
        domain_samples[domain].append((prob["index"], prob["text"][:120]))

# 統計
domain_counts = Counter(domains)

print("\n" + "="*80)
print("HLE 2500 Domain Distribution")
print("="*80)

# 実装難易度の定義
difficulty = {
    "arithmetic_basic": ("EASY", "V6で実装可能", "即座"),
    "logic": ("EASY", "V6で実装済み", "即座"),
    "number_theory_basic": ("EASY", "Executor追加で対応", "1-2時間"),
    "combinatorics": ("EASY", "Executor追加で対応", "1-2時間"),
    "probability_basic": ("MEDIUM", "Executor追加で対応", "2-4時間"),
    "geometry_basic": ("MEDIUM", "Executor追加で対応", "2-4時間"),
    "algebra_basic": ("MEDIUM", "Solver追加で対応", "4-8時間"),
    "graph_theory": ("MEDIUM", "Executor追加で対応", "4-8時間"),
    "calculus": ("HARD", "Symbolic math必要", "8-16時間"),
    "linear_algebra": ("HARD", "Matrix演算必要", "8-16時間"),
    "number_theory_advanced": ("VERY_HARD", "専門知識必要", "16+時間"),
    "probability_advanced": ("VERY_HARD", "測度論必要", "16+時間"),
    "algebra_advanced": ("VERY_HARD", "高度な理論必要", "16+時間"),
    "geometry_advanced": ("VERY_HARD", "専門知識必要", "16+時間"),
    "topology": ("VERY_HARD", "専門知識必要", "16+時間"),
    "lie_theory": ("VERY_HARD", "専門知識必要", "16+時間"),
    "algebraic_geometry": ("VERY_HARD", "専門知識必要", "16+時間"),
    "functional_analysis": ("VERY_HARD", "専門知識必要", "16+時間"),
    "other": ("UNKNOWN", "要分析", "不明"),
    "unknown": ("UNKNOWN", "要分析", "不明")
}

# ドメイン別に表示（難易度順）
difficulty_order = ["EASY", "MEDIUM", "HARD", "VERY_HARD", "UNKNOWN"]
by_difficulty = defaultdict(list)

for domain, count in domain_counts.items():
    diff_level, note, time = difficulty.get(domain, ("UNKNOWN", "不明", "不明"))
    by_difficulty[diff_level].append((domain, count, note, time))

total_by_difficulty = {}
for diff_level in difficulty_order:
    items = by_difficulty[diff_level]
    if items:
        print(f"\n{diff_level}:")
        level_total = 0
        for domain, count, note, time in sorted(items, key=lambda x: x[1], reverse=True):
            pct = count / len(problems) * 100
            print(f"  {domain:30s}: {count:4d} ({pct:5.1f}%) - {note:30s} [{time}]")
            level_total += count
        total_by_difficulty[diff_level] = level_total
        print(f"  {diff_level} Total: {level_total} ({level_total/len(problems)*100:.1f}%)")

# 累積可能性
print("\n" + "="*80)
print("Cumulative Feasibility Analysis")
print("="*80)

cumulative = 0
for diff_level in difficulty_order:
    count = total_by_difficulty.get(diff_level, 0)
    cumulative += count
    pct = cumulative / len(problems) * 100
    print(f"Up to {diff_level:12s}: {cumulative:4d} problems ({pct:5.1f}%)")

# サンプル表示
print("\n" + "="*80)
print("Sample Problems by Domain")
print("="*80)

for diff_level in difficulty_order:
    items = by_difficulty[diff_level]
    if items:
        print(f"\n{diff_level}:")
        for domain, count, note, time in sorted(items, key=lambda x: x[1], reverse=True)[:5]:
            if domain in domain_samples:
                print(f"\n  {domain} (samples):")
                for idx, text in domain_samples[domain][:2]:
                    print(f"    [{idx}] {text}...")

# 実装ロードマップ
print("\n" + "="*80)
print("Implementation Roadmap to 70%")
print("="*80)

target = int(len(problems) * 0.70)
print(f"Target: {target}/{len(problems)} problems (70%)\n")

phases = [
    ("Phase 5A", ["arithmetic_basic", "logic"], "即座", "現在実装済み"),
    ("Phase 5B", ["number_theory_basic", "combinatorics"], "2-4時間", "基本Executor追加"),
    ("Phase 5C", ["probability_basic", "geometry_basic"], "4-8時間", "中級Executor追加"),
    ("Phase 5D", ["algebra_basic", "graph_theory"], "8-16時間", "Solver追加"),
    ("Phase 5E", ["calculus", "linear_algebra"], "16-24時間", "Symbolic math"),
    ("Phase 5F", ["number_theory_advanced", "probability_advanced", "algebra_advanced"], "24-48時間", "高度な理論"),
]

cumulative_coverage = 0
for phase_name, domains_in_phase, time_est, desc in phases:
    phase_count = sum(domain_counts.get(d, 0) for d in domains_in_phase)
    cumulative_coverage += phase_count
    pct = cumulative_coverage / len(problems) * 100
    
    status = "✅" if cumulative_coverage >= target else "⏳"
    print(f"{status} {phase_name}: +{phase_count:3d} problems → {cumulative_coverage:4d} total ({pct:5.1f}%) [{time_est}]")
    print(f"   {desc}")
    
    if cumulative_coverage >= target:
        print(f"\n🎯 Target reached: {cumulative_coverage}/{len(problems)} ({pct:.1f}%) ≥ 70%")
        break

# 結果を保存
output = {
    "total": len(problems),
    "target_70pct": target,
    "domain_counts": dict(domain_counts),
    "by_difficulty": {k: v for k, v in total_by_difficulty.items()},
    "samples": {d: [(idx, text) for idx, text in samples] for d, samples in domain_samples.items()}
}

with open("hle_full_analysis.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n\nAnalysis saved to hle_full_analysis.json")
