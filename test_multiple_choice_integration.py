#!/usr/bin/env python3
"""
Multiple Choice Integration Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced


def test_multiple_choice_detection():
    """多肢選択問題の検出テスト"""
    print("=" * 80)
    print("Test: Multiple Choice Detection & Integration")
    print("=" * 80)
    print()
    
    pipeline = VerantyxV6Enhanced()
    
    # テスト問題（多肢選択）
    test_questions = [
        {
            "question": """What is the capital of France?
Answer Choices:
A. London
B. Paris
C. Berlin
D. Rome""",
            "expected": "B"
        },
        {
            "question": """What is 2 + 2?
A. 3
B. 4
C. 5
D. 6""",
            "expected": "B"
        },
        {
            "question": """Which of the following is a prime number?
A. 4
B. 6
C. 7
D. 8""",
            "expected": "C"
        }
    ]
    
    passed = 0
    for i, test in enumerate(test_questions, 1):
        print(f"\nTest {i}:")
        print(f"Question: {test['question'][:60]}...")
        
        try:
            result = pipeline.solve(test['question'])
            answer = result.get('answer')
            status = result.get('status')
            
            print(f"  Answer: {answer}")
            print(f"  Expected: {test['expected']}")
            print(f"  Status: {status}")
            
            # IR確認
            if 'ir' in result:
                ir = result['ir']
                print(f"  IR Domain: {ir.get('domain')}")
                print(f"  IR Answer Schema: {ir.get('answer_schema')}")
            
            # 正解判定
            if str(answer).strip().upper() == str(test['expected']).strip().upper():
                print(f"  ✅ CORRECT")
                passed += 1
            else:
                print(f"  ❌ INCORRECT")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(test_questions)}")
    print("=" * 80)


if __name__ == "__main__":
    test_multiple_choice_detection()
