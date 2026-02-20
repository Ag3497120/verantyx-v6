"""
Phase 5E Test: Linear Algebra & Calculus
Target: 70% (7/10)
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced


def run_test():
    """Phase 5Eテスト実行"""
    
    # テストケース: 線形代数5問 + 微積分5問
    test_cases = [
        # Linear Algebra (5 problems)
        {
            "question": "Find the determinant of matrix [[2, 3], [1, 4]].",
            "expected": 5,
            "category": "linear_algebra"
        },
        {
            "question": "Calculate the dot product of vectors [1, 2, 3] and [4, 5, 6].",
            "expected": 32,
            "category": "linear_algebra"
        },
        {
            "question": "What is the determinant of a 3x3 identity matrix?",
            "expected": 1,
            "category": "linear_algebra"
        },
        {
            "question": "Find the dot product of [2, 0] and [0, 3].",
            "expected": 0,
            "category": "linear_algebra"
        },
        {
            "question": "Calculate the determinant of [[1, 2], [3, 4]].",
            "expected": -2,
            "category": "linear_algebra"
        },
        
        # Calculus (5 problems)
        {
            "question": "What is the derivative of x^2 with respect to x?",
            "expected": "2x",
            "category": "calculus"
        },
        {
            "question": "Find the derivative of 3x^3.",
            "expected": "9x^2",
            "category": "calculus"
        },
        {
            "question": "What is the integral of 2x with respect to x?",
            "expected": "x^2",
            "category": "calculus"
        },
        {
            "question": "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2.",
            "expected": 4,
            "category": "calculus"
        },
        {
            "question": "What is the derivative of 5x?",
            "expected": 5,
            "category": "calculus"
        }
    ]
    
    pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
    
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("Phase 5E Test: Linear Algebra & Calculus")
    print("=" * 80)
    print()
    
    for i, test in enumerate(test_cases, 1):
        question = test["question"]
        expected = test["expected"]
        category = test["category"]
        
        print(f"[Test {i}/10] {category} - {question}")
        
        result = pipeline.solve(question)
        answer = result.get("answer")
        
        # 型に応じた比較
        success = False
        if isinstance(expected, (int, float)):
            # 期待値が数値の場合
            if isinstance(answer, (int, float)):
                success = abs(answer - expected) < 0.01
            elif isinstance(answer, str):
                # 文字列を数値に変換試行
                try:
                    answer_num = float(answer)
                    success = abs(answer_num - expected) < 0.01
                except:
                    success = False
        elif isinstance(expected, str):
            if isinstance(answer, str):
                # 文字列比較（空白・*・^/**を正規化）
                exp_norm = expected.replace(" ", "").replace("*", "").replace("^", "").lower()
                ans_norm = answer.replace(" ", "").replace("*", "").replace("^", "").lower()
                success = ans_norm == exp_norm
            elif isinstance(answer, (int, float)):
                # 答えが数値の場合、期待値を数値に変換試行
                try:
                    expected_num = float(expected)
                    success = abs(answer - expected_num) < 0.01
                except:
                    success = False
        else:
            success = answer == expected
        
        if success:
            print(f"✅ PASS: {answer}")
            passed += 1
        else:
            print(f"❌ FAIL: {answer} (expected {expected})")
            failed += 1
        
        print()
    
    print("=" * 80)
    accuracy = (passed / len(test_cases)) * 100
    print(f"Results: {passed}/{len(test_cases)} passed ({accuracy:.1f}%)")
    print(f"Target: 7/10 (70%)")
    
    if accuracy >= 70:
        print("✅ Phase 5E Test PASSED")
    else:
        print("❌ Phase 5E Test FAILED")
    
    print("=" * 80)
    
    return accuracy >= 70


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
