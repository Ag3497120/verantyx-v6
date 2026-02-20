"""
AVH Enhanced Pipeline Test

統合パイプラインでの論理問題テスト
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_avh_enhanced import VerantyxAvhEnhanced


def test_logic_problems():
    """論理問題のテスト"""
    print("="*80)
    print("AVH Enhanced Pipeline Test - Logic Problems")
    print("="*80)
    print()
    
    # 統合パイプライン初期化
    pipeline = VerantyxAvhEnhanced(use_avh=True)
    
    # テスト問題（論理）
    test_cases = [
        {
            "problem": "Is '(A | ~A)' a tautology?",
            "expected": True,
            "name": "Law of Excluded Middle"
        },
        {
            "problem": "Is '((A -> B) & A) -> B' always true?",
            "expected": True,
            "name": "Modus Ponens"
        },
        {
            "problem": "Is 'A & ~A' satisfiable?",
            "expected": False,
            "name": "Contradiction"
        },
        {
            "problem": "Check if 'A -> A' is a tautology",
            "expected": True,
            "name": "Identity"
        },
        {
            "problem": "Is '(A -> B) -> (~B -> ~A)' valid?",
            "expected": True,
            "name": "Contrapositive"
        }
    ]
    
    results = {
        "total": len(test_cases),
        "passed": 0,
        "avh_used": 0,
        "verantyx_only": 0
    }
    
    for i, test in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test['name']} ---")
        print(f"Problem: {test['problem']}")
        
        try:
            result = pipeline.solve(test['problem'])
            
            print(f"Answer: {result.answer}")
            print(f"Status: {result.status}")
            print(f"Reasoning Mode: {result.reasoning_mode}")
            print(f"Verified: {result.verified}")
            print(f"Confidence: {result.confidence:.2f}")
            
            if result.axioms_used:
                print(f"Axioms Used: {result.axioms_used[:3]}")
            
            # 正解判定（簡易版）
            answer_str = str(result.answer).lower()
            expected_str = str(test['expected']).lower()
            
            is_correct = (
                answer_str == expected_str or
                (expected_str == "true" and answer_str in ["true", "yes", "tautology"]) or
                (expected_str == "false" and answer_str in ["false", "no", "not satisfiable"])
            )
            
            if is_correct:
                results["passed"] += 1
                print("✅ CORRECT")
            else:
                print(f"❌ INCORRECT (expected {test['expected']})")
            
            # モード集計
            if result.reasoning_mode == "avh_cross":
                results["avh_used"] += 1
            elif result.reasoning_mode == "verantyx_only":
                results["verantyx_only"] += 1
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # サマリー
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']} ({results['passed']/results['total']*100:.1f}%)")
    print(f"AVH Cross used: {results['avh_used']}")
    print(f"Verantyx only: {results['verantyx_only']}")
    print("="*80)
    
    return results


def test_non_logic_problems():
    """非論理問題のテスト（Verantyxフォールバック）"""
    print("\n" + "="*80)
    print("AVH Enhanced Pipeline Test - Non-Logic Problems")
    print("="*80)
    print()
    
    pipeline = VerantyxAvhEnhanced(use_avh=True)
    
    test_cases = [
        {"problem": "Calculate 15 + 27", "name": "Arithmetic"},
        {"problem": "What is 5 factorial?", "name": "Factorial"},
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test['name']} ---")
        print(f"Problem: {test['problem']}")
        
        try:
            result = pipeline.solve(test['problem'])
            
            print(f"Answer: {result.answer}")
            print(f"Status: {result.status}")
            print(f"Reasoning Mode: {result.reasoning_mode}")
            print(f"Confidence: {result.confidence:.2f}")
            print()
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AVH ENHANCED PIPELINE TEST SUITE")
    print("="*80)
    print()
    
    # Test 1: Logic problems (AVH should be used)
    logic_results = test_logic_problems()
    
    # Test 2: Non-logic problems (Verantyx fallback)
    test_non_logic_problems()
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Logic Problems: {logic_results['passed']}/{logic_results['total']} correct")
    print(f"AVH Integration: {'✅ WORKING' if logic_results['avh_used'] > 0 else '❌ NOT USED'}")
    print("="*80)
