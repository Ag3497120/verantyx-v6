"""
Phase 5D テスト：代数基本・グラフ理論
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced

# テストケース
test_cases = [
    # 代数基本
    {"question": "Solve for x: 2x + 3 = 11", "expected": "4", "domain": "algebra"},
    {"question": "What is the value of x in the equation 3x - 5 = 10?", "expected": "5", "domain": "algebra"},
    {"question": "Simplify: (x + 2)(x - 3)", "expected": "x^2 - x - 6", "domain": "algebra"},
    {"question": "Factor: x^2 + 5x + 6", "expected": "(x + 2)(x + 3)", "domain": "algebra"},
    {"question": "Evaluate: x^2 + 3x when x = 2", "expected": "10", "domain": "algebra"},
    
    # グラフ理論
    {"question": "How many edges are in a complete graph with 4 vertices (K4)?", "expected": "6", "domain": "graph"},
    {"question": "What is the minimum number of edges needed to connect 5 vertices in a tree?", "expected": "4", "domain": "graph"},
    {"question": "Is a graph with 3 vertices and 3 edges necessarily cyclic?", "expected": "True", "domain": "graph"},
    {"question": "What is the degree sum of all vertices in a graph with 5 edges?", "expected": "10", "domain": "graph"},
    {"question": "How many vertices does a binary tree of height 2 have (including root)?", "expected": "7", "domain": "graph"},
]

def run_tests():
    v6 = VerantyxV6Enhanced(use_beam_search=False)  # Greedy
    
    print("=" * 80)
    print("Phase 5D Test: Algebra & Graph Theory")
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
            
            # 文字列比較（大小文字・空白を無視）
            is_correct = False
            if answer is not None and expected is not None:
                answer_str = str(answer).strip().lower().replace(" ", "")
                expected_str = str(expected).strip().lower().replace(" ", "")
                is_correct = answer_str == expected_str
            
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
        print("✅ Phase 5D Test PASSED")
        return 0
    else:
        print("❌ Phase 5D Test FAILED")
        return 1
    print("=" * 80)

if __name__ == "__main__":
    exit(run_tests())
