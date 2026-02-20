"""
Phase 5B テスト：数論・組み合わせ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced

# テストケース
test_cases = [
    # 数論
    {"question": "Is 17 a prime number?", "expected": "True", "domain": "number_theory"},
    {"question": "What is the GCD of 48 and 18?", "expected": "6", "domain": "number_theory"},
    {"question": "Calculate 5 factorial (5!)", "expected": "120", "domain": "number_theory"},
    {"question": "How many divisors does 12 have?", "expected": "6", "domain": "number_theory"},
    {"question": "Find the LCM of 12 and 15", "expected": "60", "domain": "number_theory"},
    
    # 組み合わせ
    {"question": "Calculate P(5, 2) - the number of permutations", "expected": "20", "domain": "combinatorics"},
    {"question": "Calculate C(6, 2) - the number of combinations", "expected": "15", "domain": "combinatorics"},
    {"question": "What is the binomial coefficient C(10, 3)?", "expected": "120", "domain": "combinatorics"},
    {"question": "How many ways can you arrange 4 items from 6?", "expected": "360", "domain": "combinatorics"},
    {"question": "In how many ways can you choose 3 items from 5?", "expected": "10", "domain": "combinatorics"},
]

def run_tests():
    v6 = VerantyxV6Enhanced(use_beam_search=False)  # Use Greedy instead of BeamSearch
    
    passed = 0
    failed = 0
    
    print("="*80)
    print("Phase 5B Test: Number Theory & Combinatorics")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['domain']}")
        print(f"Q: {test['question']}")
        print(f"Expected: {test['expected']}")
        
        result = v6.solve(test['question'], use_crystal=False)
        
        # 数値比較
        try:
            ans_num = float(result.get('answer', 0))
            exp_num = float(test['expected'])
            match = abs(ans_num - exp_num) < 0.01
        except:
            match = str(result.get('answer')) == test['expected']
        
        if match:
            print(f"✅ PASS: {result.get('answer')}")
            passed += 1
        else:
            print(f"❌ FAIL: {result.get('answer')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Trace (last 5): {result.get('trace', [])[-5:]}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"Target: 7/10 (70%)")
    
    if passed >= 7:
        print("✅ Phase 5B Test PASSED")
        result_status = True
    else:
        print("❌ Phase 5B Test FAILED")
        result_status = False
    
    print(f"{'='*80}")
    
    return result_status

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
