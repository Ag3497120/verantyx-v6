"""
Phase 5F + 5G テスト

新しいexecutorとピースの動作確認
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced


def test_phase_5f_5g():
    """Phase 5F + 5G の簡易テスト"""
    pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
    
    tests = [
        # Phase 5F: Advanced Combinatorics
        {
            "question": "What is the Stirling number of the second kind S(5, 3)?",
            "expected": 25,
            "category": "Advanced Combinatorics"
        },
        {
            "question": "Calculate the Bell number B(4).",
            "expected": 15,
            "category": "Advanced Combinatorics"
        },
        {
            "question": "What is the 4th Catalan number C(4)?",
            "expected": 14,
            "category": "Advanced Combinatorics"
        },
        
        # Phase 5G: Multiple Choice
        {
            "question": "Which of the following is a prime number?\n\nAnswer Choices:\nA. 15\nB. 17\nC. 18\nD. 20",
            "expected": "B",
            "category": "Multiple Choice"
        },
        
        # Phase 5G: String Operations
        {
            "question": "What is the length of the string 'Hello, World!'?",
            "expected": 13,
            "category": "String Operations"
        },
        {
            "question": "Is the word 'racecar' a palindrome?",
            "expected": "Yes",
            "category": "String Operations"
        },
        
        # Modular Arithmetic
        {
            "question": "Calculate 7^10 mod 13",
            "expected": 4,
            "category": "Modular Arithmetic"
        },
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(tests)}: {test['category']}")
        print(f"Question: {test['question'][:80]}...")
        print(f"Expected: {test['expected']}")
        
        try:
            result = pipeline.solve(test['question'])
            answer = result.get('answer')
            status = result.get('status')
            
            # 簡易的な正解判定
            is_correct = False
            if answer is not None:
                if isinstance(answer, (int, float)) and isinstance(test['expected'], (int, float)):
                    is_correct = abs(answer - test['expected']) < 0.01
                else:
                    is_correct = str(answer).strip().lower() == str(test['expected']).strip().lower()
            
            print(f"Got: {answer}")
            print(f"Status: {status}")
            print(f"Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
            
            results.append({
                'test': i,
                'category': test['category'],
                'correct': is_correct,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'test': i,
                'category': test['category'],
                'correct': False,
                'status': 'ERROR'
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    category_stats = {}
    for r in results:
        cat = r['category']
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'correct': 0}
        category_stats[cat]['total'] += 1
        if r['correct']:
            category_stats[cat]['correct'] += 1
    
    for cat, stats in category_stats.items():
        acc = stats['correct'] / stats['total'] * 100
        print(f"{cat:30s} {stats['correct']}/{stats['total']} ({acc:.0f}%)")
    
    total_correct = sum(r['correct'] for r in results)
    total = len(results)
    print(f"\n{'='*80}")
    print(f"Total: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_phase_5f_5g()
