#!/usr/bin/env python3
"""
Quick HLE Test - Priority 1完了後の効果測定
"""
import sys
import json
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


def test_quick_sample():
    """HLEから50問サンプルテスト"""
    pipeline = VerantyxV6Enhanced()
    
    # HLEデータをロード
    with open('hle_2500_eval.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    # 最初の50問をテスト
    sample_size = 50
    correct = 0
    failed = 0
    
    print("=" * 80)
    print(f"Quick HLE Test ({sample_size} questions)")
    print("=" * 80)
    print()
    
    for i, q in enumerate(questions[:sample_size]):
        question = q['question']
        expected = q['answer']
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{sample_size}")
        
        try:
            result = pipeline.solve(question)
            answer = result.get('answer')
            
            # flexible_matchで判定
            if answer is not None and expected is not None:
                if flexible_match(answer, expected, tolerance=1e-4):
                    correct += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Error on Q{i+1}: {e}")
    
    accuracy = (correct / sample_size) * 100
    
    print()
    print("=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Correct: {correct}/{sample_size}")
    print(f"Failed: {failed}/{sample_size}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    print(f"Baseline (before Priority 1): 3.5%")
    print(f"Improvement: {accuracy - 3.5:+.1f} points")
    print("=" * 80)


if __name__ == "__main__":
    test_quick_sample()
