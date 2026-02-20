#!/usr/bin/env python3
"""
HLE 200問テスト - Phase 5G実装効果測定
"""
import sys
import json
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


def test_hle_200():
    """HLE 200問テスト"""
    pipeline = VerantyxV6Enhanced()
    
    # HLEデータをロード
    with open('hle_2500_eval.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    sample_size = 200
    correct = 0
    failed = 0
    
    print("=" * 80)
    print(f"HLE {sample_size} Question Test")
    print("=" * 80)
    print()
    
    for i, q in enumerate(questions[:sample_size]):
        question = q['question']
        expected = q['answer']
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{sample_size} ({correct} correct so far)")
        
        try:
            result = pipeline.solve(question)
            answer = result.get('answer')
            
            # flexible_matchで判定
            if answer is not None and expected is not None:
                if flexible_match(answer, expected, tolerance=1e-4):
                    correct += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Error on Q{i+1}: {e}")
    
    accuracy = (correct / sample_size) * 100
    
    print()
    print("=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Correct: {correct}/{sample_size}")
    print(f"Failed: {failed}/{sample_size}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print(f"Baseline (before Phase 5G): 3.5%")
    print(f"Improvement: {accuracy - 3.5:+.2f} points")
    print()
    
    # 目標との比較
    if accuracy >= 10.0:
        print("🎉 目標10%達成！")
    elif accuracy >= 7.0:
        print("📈 大幅改善（2倍以上）")
    elif accuracy >= 5.0:
        print("📊 顕著な改善")
    else:
        print("⚠️  さらなる改善が必要")
    
    print("=" * 80)


if __name__ == "__main__":
    test_hle_200()
