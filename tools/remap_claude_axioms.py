#!/usr/bin/env python3
"""
Claude公理の再マッピング

knowledge.axiom → 実行可能なexecutorにマッピング
"""
import json
from pathlib import Path

# ドメイン → Executor マッピング
DOMAIN_EXECUTOR_MAP = {
    "algebra": "executors.algebra.evaluate_expression",
    "linear_algebra": "executors.linear_algebra.matrix_multiply",
    "calculus": "executors.calculus.derivative",
    "geometry": "executors.geometry.calculate_area",
    "number_theory": "executors.number_theory.gcd",
    "combinatorics": "executors.combinatorics.combination",
    "probability": "executors.probability.calculate_probability",
    "statistics": "executors.statistics.mean",
    "physics": "executors.arithmetic.evaluate",
    "chemistry": "executors.arithmetic.evaluate",
    "logic": "executors.logic.prop_truth_table"
}

def remap_axioms(input_file: str, output_file: str):
    """
    Claude公理を実行可能なexecutorに再マッピング
    
    Args:
        input_file: pieces_claude.json
        output_file: pieces_claude_remapped.json
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    remapped_count = 0
    unknown_domains = set()
    
    for piece in data['pieces']:
        domain = piece.get('domain', '')
        
        if piece['executor'] == 'knowledge.axiom':
            # ドメインに応じてexecutorをマッピング
            if domain in DOMAIN_EXECUTOR_MAP:
                piece['executor'] = DOMAIN_EXECUTOR_MAP[domain]
                remapped_count += 1
            else:
                # 不明なドメインはarithmeticにフォールバック
                piece['executor'] = 'executors.arithmetic.evaluate'
                remapped_count += 1
                unknown_domains.add(domain)
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Remapped {remapped_count}/{len(data['pieces'])} pieces")
    print(f"  knowledge.axiom → domain-specific executors")
    
    if unknown_domains:
        print(f"\nUnknown domains (mapped to arithmetic):")
        for domain in sorted(unknown_domains):
            print(f"  - {domain}")
    
    print(f"\nSaved to: {output_file}")
    print(f"Size: {Path(output_file).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / "pieces" / "pieces_claude.json"
    output_file = Path(__file__).parent.parent / "pieces" / "pieces_claude_remapped.json"
    
    print("=" * 70)
    print("Claude Axiom Remapping")
    print("=" * 70)
    
    remap_axioms(str(input_file), str(output_file))
    
    print("=" * 70)
    print("✅ Remapping complete!")
    print("=" * 70)
