#!/usr/bin/env python3
"""
Priority 3 Test - 問題タイプ検出の効果測定
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced


def test_problem_type_detection():
    """問題タイプ検出のテスト"""
    pipeline = VerantyxV6Enhanced()
    
    test_cases = [
        {
            "question": "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Rome",
            "expected_type": "multiple_choice"
        },
        {
            "question": "Solve for x: 2x + 3 = 7",
            "expected_type": "equation"
        },
        {
            "question": "Calculate 25 + 37",
            "expected_type": "calculation"
        },
        {
            "question": "How many ways can you arrange 5 books on a shelf?",
            "expected_type": "counting"
        },
        {
            "question": "What is the area of a circle with radius 5?",
            "expected_type": "geometry"
        }
    ]
    
    print("=" * 80)
    print("Priority 3 Test: Problem Type Detection")
    print("=" * 80)
    print()
    
    passed = 0
    for i, test in enumerate(test_cases, 1):
        question = test['question']
        expected_type = test['expected_type']
        
        print(f"Test {i}: {question[:60]}...")
        
        try:
            result = pipeline.solve(question)
            
            # IRのメタデータから問題タイプを確認
            ir = result.get('ir', {})
            metadata = ir.get('metadata', {})
            detected_type = metadata.get('problem_type', 'unknown')
            confidence = metadata.get('problem_type_confidence', 0.0)
            domain = ir.get('domain', 'unknown')
            
            print(f"  Detected Type: {detected_type} (confidence: {confidence:.2f})")
            print(f"  Domain: {domain}")
            print(f"  Expected: {expected_type}")
            
            if detected_type == expected_type:
                print(f"  ✅ CORRECT")
                passed += 1
            else:
                print(f"  ⚠️  Type mismatch, but checking domain boost...")
                # ドメインが適切なら部分的に成功
                if domain in expected_type or expected_type in domain:
                    print(f"  ✅ PARTIAL (domain boost working)")
                    passed += 0.5
                else:
                    print(f"  ❌ INCORRECT")
                    
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    print("=" * 80)
    print(f"Passed: {passed}/{len(test_cases)}")
    print("=" * 80)


if __name__ == "__main__":
    test_problem_type_detection()
