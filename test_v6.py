"""
Verantyx V6 テストスクリプト

基本動作確認用
"""

import sys
import os

# パスを追加
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import VerantyxV6


def test_basic():
    """基本動作テスト"""
    print("=" * 80)
    print("Verantyx V6 - Basic Test")
    print("=" * 80)
    print()
    
    # V6初期化
    v6 = VerantyxV6(use_beam_search=False)  # Greedy（高速）
    
    # テスト問題
    test_problems = [
        {
            "text": "What is 1 + 1?",
            "expected": "2",
            "note": "Arithmetic (simple)"
        },
        {
            "text": "Calculate 5 * 6.",
            "expected": "30",
            "note": "Arithmetic (multiplication)"
        },
        {
            "text": "Is 10 greater than 5?",
            "expected": "True",
            "note": "Boolean (comparison)"
        },
        {
            "text": """Which is correct?
            
            Answer Choices:
            A. 2 + 2 = 4
            B. 2 + 2 = 5
            C. 2 + 2 = 6""",
            "expected": "A",
            "note": "Multiple-choice"
        },
        {
            "text": "How many vertices does a triangle have?",
            "expected": "3",
            "note": "Geometry (count)"
        }
    ]
    
    results = []
    for i, problem in enumerate(test_problems):
        print(f"\n[Test {i+1}/{len(test_problems)}] {problem['note']}")
        print(f"Question: {problem['text'][:80]}...")
        print(f"Expected: {problem['expected']}")
        
        result = v6.solve(problem["text"], problem["expected"])
        
        print(f"Status: {result['status']}")
        print(f"Answer: {result.get('answer', 'N/A')}")
        
        if result["status"] == "VERIFIED":
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            print(f"Trace: {result.get('trace', [])[-5:]}")  # 最後の5ステップ
        
        results.append(result)
    
    # 統計
    print()
    v6.print_stats()
    
    # サマリー
    verified = sum(1 for r in results if r["status"] == "VERIFIED")
    print(f"\nTest Results: {verified}/{len(results)} VERIFIED")
    
    return results


def test_ir_extraction():
    """IR抽出テスト"""
    print("\n" + "=" * 80)
    print("IR Extraction Test")
    print("=" * 80)
    print()
    
    from decomposer.decomposer import RuleBasedDecomposer
    
    decomposer = RuleBasedDecomposer()
    
    test_texts = [
        "What is 1 + 1?",
        "Is p -> p a tautology?",
        "Find the smallest prime number greater than 10.",
        "How many edges does a complete graph with 5 vertices have?",
        "Black to move. What is mate in 2?"
    ]
    
    for text in test_texts:
        print(f"Text: {text}")
        ir = decomposer.decompose(text)
        print(f"  Task: {ir.task.value}")
        print(f"  Domain: {ir.domain.value}")
        print(f"  Answer Schema: {ir.answer_schema.value}")
        print(f"  Entities: {len(ir.entities)}")
        print(f"  Options: {len(ir.options)}")
        print()


def test_piece_search():
    """ピース検索テスト"""
    print("\n" + "=" * 80)
    print("Piece Search Test")
    print("=" * 80)
    print()
    
    from pieces.piece import PieceDB
    from decomposer.decomposer import RuleBasedDecomposer
    
    piece_db_path = os.path.join(os.path.dirname(__file__), "pieces/piece_db.jsonl")
    piece_db = PieceDB(piece_db_path)
    decomposer = RuleBasedDecomposer()
    
    print(f"Loaded {len(piece_db.pieces)} pieces")
    print()
    
    test_texts = [
        "What is 1 + 1?",
        "Is this statement true?",
        "Choose the correct answer: A, B, or C"
    ]
    
    for text in test_texts:
        print(f"Text: {text}")
        ir = decomposer.decompose(text)
        ir_dict = ir.to_dict()
        
        matches = piece_db.search(ir_dict, top_k=3)
        print(f"  Top matches:")
        for piece, score in matches:
            print(f"    - {piece.piece_id} (score={score:.2f})")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verantyx V6 Test")
    parser.add_argument("--test", choices=["basic", "ir", "piece", "all"], default="all",
                        help="Test type")
    
    args = parser.parse_args()
    
    if args.test in ["basic", "all"]:
        test_basic()
    
    if args.test in ["ir", "all"]:
        test_ir_extraction()
    
    if args.test in ["piece", "all"]:
        test_piece_search()
