#!/usr/bin/env python3
"""
search_keywords() の識別力診断テスト

全文平均（search）と キーワード絞り込み（search_keywords）を比較。
discriminative power が向上しているか確認する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge.concept_search import ConceptSearcher

cs = ConceptSearcher(verbose=True)

CASES = [
    {
        "label": "calculus",
        "full_text": "What is the derivative of sin(x) with respect to x?",
        "keywords": ["derivative", "sin"],
        "expected_top": ["calculus", "algebra"],
    },
    {
        "label": "combinatorics",
        "full_text": "How many ways to arrange all permutations of 4 objects?",
        "keywords": ["permutation", "factorial"],
        "expected_top": ["combinatorics", "number_theory"],
    },
    {
        "label": "linear_algebra",
        "full_text": "Find the eigenvalues of the 2x2 matrix [[1,2],[3,4]]",
        "keywords": ["eigenvalue", "matrix"],
        "expected_top": ["linear_algebra", "algebra"],
    },
    {
        "label": "probability",
        "full_text": "What is the probability of rolling a 6 on a fair die twice in a row?",
        "keywords": ["probability", "dice"],
        "expected_top": ["probability", "statistics"],
    },
    {
        "label": "number_theory",
        "full_text": "What is the GCD of 48 and 18?",
        "keywords": ["gcd", "prime", "divisor"],
        "expected_top": ["number_theory", "algebra"],
    },
    {
        "label": "chemistry",
        "full_text": "What is the molecular weight of H2SO4?",
        "keywords": ["chemistry", "molecular"],
        "expected_top": ["chemistry"],
    },
]

print("=" * 70)
print("全文 vs キーワード 識別力比較")
print("=" * 70)

for case in CASES:
    print(f"\n[{case['label']}]")
    print(f"  Full text: {case['full_text'][:60]}")
    print(f"  Keywords:  {case['keywords']}")

    # 全文検索
    full_scores = cs.search(case["full_text"], top_k=20)
    top_full = list(full_scores.items())[:3]

    # キーワード検索
    kw_scores = cs.search_keywords(case["keywords"], top_k=20)
    top_kw = list(kw_scores.items())[:3]

    print(f"  Full top3:    {top_full}")
    print(f"  Keyword top3: {top_kw}")

    # 識別力評価：期待するドメインがtop3に入っているか
    full_domains = [d for d, _ in top_full]
    kw_domains   = [d for d, _ in top_kw]
    full_hit = any(d in full_domains for d in case["expected_top"])
    kw_hit   = any(d in kw_domains   for d in case["expected_top"])

    print(f"  Expected: {case['expected_top']}")
    print(f"  Full hit: {'✅' if full_hit else '❌'}  Keyword hit: {'✅' if kw_hit else '❌'}")

print("\n" + "=" * 70)
print("完了")
