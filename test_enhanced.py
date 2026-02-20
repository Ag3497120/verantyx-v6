"""
Verantyx V6 Enhanced テストスクリプト

本来の構想準拠版のテスト
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced


def test_enhanced_pipeline():
    """強化パイプラインのテスト"""
    print("=" * 80)
    print("Verantyx V6 Enhanced - Test (本来の構想準拠)")
    print("=" * 80)
    print()
    
    # V6 Enhanced初期化
    v6 = VerantyxV6Enhanced(
        use_beam_search=False,  # Greedy（高速）
        use_simulation=True     # Crossシミュレーション有効
    )
    
    # テスト問題
    test_problems = [
        {
            "text": "What is 1 + 1?",
            "expected": "2",
            "note": "Arithmetic (simple)"
        },
        {
            "text": "Calculate 5 * 6.",
            "expected": "30",
            "note": "Arithmetic (multiplication)"
        },
        {
            "text": "Is p -> p a tautology?",
            "expected": "True",
            "note": "Propositional Logic (with simulation)"
        },
        {
            "text": """Which is correct?
            
            Answer Choices:
            A. 2 + 2 = 4
            B. 2 + 2 = 5
            C. 2 + 2 = 6""",
            "expected": "A",
            "note": "Multiple-choice"
        },
        {
            "text": "How many vertices does a triangle have?",
            "expected": "3",
            "note": "Geometry (count)"
        }
    ]
    
    results = []
    for i, problem in enumerate(test_problems):
        print(f"\n[Test {i+1}/{len(test_problems)}] {problem['note']}")
        print(f"Question: {problem['text'][:80]}...")
        print(f"Expected: {problem['expected']}")
        
        result = v6.solve(problem["text"], problem["expected"], use_crystal=False)
        
        print(f"Status: {result['status']}")
        print(f"Answer: {result.get('answer', 'N/A')}")
        
        # Crossシミュレーションの結果を表示
        if "simulation" in result:
            sim = result["simulation"]
            print(f"Simulation: {sim.get('method', 'N/A')} - {sim.get('status', 'N/A')}")
        
        if result["status"] == "VERIFIED":
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            # トレース（最後の10ステップ）
            trace = result.get('trace', [])
            if len(trace) > 10:
                print(f"Trace (last 10): {trace[-10:]}")
            else:
                print(f"Trace: {trace}")
        
        results.append(result)
    
    # 統計
    print()
    v6.print_stats()
    
    # サマリー
    verified = sum(1 for r in results if r["status"] == "VERIFIED")
    print(f"\nTest Results: {verified}/{len(results)} VERIFIED")
    
    return results


if __name__ == "__main__":
    test_enhanced_pipeline()
