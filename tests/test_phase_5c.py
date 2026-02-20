"""
Phase 5C テスト：確率・幾何基本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced

# テストケース
test_cases = [
    # 確率
    {"question": "What is the probability of flipping heads on a fair coin?", "expected": "0.5", "domain": "probability"},
    {"question": "What is the probability of rolling a 6 on a standard die?", "expected": "0.166667", "domain": "probability"},
    {"question": "If you flip a coin twice, what is the probability of getting two heads?", "expected": "0.25", "domain": "probability"},
    {"question": "What is the probability of drawing a heart from a standard deck of 52 cards?", "expected": "0.25", "domain": "probability"},
    {"question": "Calculate the expected value of rolling a fair 6-sided die", "expected": "3.5", "domain": "probability"},
    
    # 幾何
    {"question": "What is the area of a circle with radius 5?", "expected": "78.54", "domain": "geometry"},
    {"question": "Calculate the area of a triangle with base 10 and height 6", "expected": "30", "domain": "geometry"},
    {"question": "What is the circumference of a circle with radius 7?", "expected": "43.98", "domain": "geometry"},
    {"question": "If a right triangle has legs of length 3 and 4, what is the hypotenuse?", "expected": "5", "domain": "geometry"},
    {"question": "What is the perimeter of a rectangle with length 8 and width 5?", "expected": "26", "domain": "geometry"},
]

def run_tests():
    v6 = VerantyxV6Enhanced(use_beam_search=False)  # Use Greedy (proven better in 5B)
    
    print("=" * 80)
    print("Phase 5C Test: Probability & Geometry")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        domain = test_case["domain"]
        
        print(f"[Test {i}/{len(test_cases)}] {domain}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        
        try:
            result = v6.solve(question, expected_answer=expected, use_crystal=False)
            answer = result.get("answer")
            status = result.get("status", "UNKNOWN")
            trace = result.get("trace", [])
            
            # 数値比較（tolerance付き）
            is_correct = False
            try:
                expected_float = float(expected)
                answer_float = float(answer) if answer is not None else None
                if answer_float is not None:
                    # 1%の誤差を許容
                    tolerance = max(0.01, abs(expected_float) * 0.01)
                    is_correct = abs(answer_float - expected_float) < tolerance
            except:
                is_correct = str(answer) == str(expected)
            
            if is_correct:
                print(f"✅ PASS: {answer}")
                passed += 1
            else:
                print(f"❌ FAIL: {answer}")
                print(f"   Status: {status}")
                print(f"   Trace (last 5): {trace[-5:]}")
                failed += 1
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
        
        print()
    
    print("=" * 80)
    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"Target: {int(len(test_cases)*0.7)}/{len(test_cases)} (70%)")
    
    if passed >= int(len(test_cases) * 0.7):
        print("✅ Phase 5C Test PASSED")
        return 0
    else:
        print("❌ Phase 5C Test FAILED")
        return 1
    print("=" * 80)

if __name__ == "__main__":
    exit(run_tests())
