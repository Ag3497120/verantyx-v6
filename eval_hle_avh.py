"""
HLE 2500 Evaluation with AVH Integration

AVH統合パイプラインでHLE 2500問を評価
"""
import sys
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_avh_enhanced import VerantyxAvhEnhanced


def load_hle_2500(file_path: str = "hle_2500_eval.jsonl") -> List[Dict[str, Any]]:
    """HLE 2500データをロード"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def evaluate_hle(
    pipeline: VerantyxAvhEnhanced,
    questions: List[Dict[str, Any]],
    start_idx: int = 0,
    batch_size: int = 100,
    save_interval: int = 50
) -> Dict[str, Any]:
    """
    HLE評価
    
    Args:
        pipeline: 統合パイプライン
        questions: 問題リスト
        start_idx: 開始インデックス
        batch_size: バッチサイズ
        save_interval: 保存間隔
    
    Returns:
        評価結果
    """
    results = {
        "total": len(questions),
        "evaluated": 0,
        "correct": 0,
        "incorrect": 0,
        "failed": 0,
        "avh_used": 0,
        "verantyx_only": 0,
        "results": []
    }
    
    end_idx = min(start_idx + batch_size, len(questions))
    
    print("=" * 80)
    print(f"Evaluating questions {start_idx+1} to {end_idx}")
    print("=" * 80)
    print()
    
    for idx in range(start_idx, end_idx):
        q = questions[idx]
        q_id = q['id']
        question = q['question']
        expected_answer = q['answer']
        category = q.get('category', 'Unknown')
        
        # 進捗表示
        if (idx - start_idx + 1) % 10 == 0:
            print(f"Progress: {idx - start_idx + 1}/{end_idx - start_idx} ({(idx - start_idx + 1)/(end_idx - start_idx)*100:.1f}%)")
        
        try:
            # 評価実行
            start_time = time.time()
            result = pipeline.solve(question)
            elapsed = time.time() - start_time
            
            answer = result.answer
            status = result.status
            reasoning_mode = result.reasoning_mode
            
            # 正解判定（簡易版）
            is_correct = False
            if answer is not None and expected_answer is not None:
                # 型に応じた比較
                if isinstance(answer, str) and isinstance(expected_answer, str):
                    # 大文字小文字無視、空白除去
                    is_correct = answer.strip().lower() == expected_answer.strip().lower()
                elif isinstance(answer, (int, float)) and isinstance(expected_answer, (int, float)):
                    is_correct = abs(answer - expected_answer) < 0.01
                else:
                    is_correct = str(answer).strip().lower() == str(expected_answer).strip().lower()
            
            # 結果記録
            item_result = {
                "id": q_id,
                "category": category,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "expected": expected_answer,
                "answer": answer,
                "correct": is_correct,
                "status": status,
                "reasoning_mode": reasoning_mode,
                "elapsed": round(elapsed, 2)
            }
            
            results["results"].append(item_result)
            results["evaluated"] += 1
            
            if is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1
            
            # モード集計
            if reasoning_mode == "avh_cross":
                results["avh_used"] += 1
            elif reasoning_mode == "verantyx_only":
                results["verantyx_only"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["results"].append({
                "id": q_id,
                "category": category,
                "error": str(e),
                "status": "ERROR"
            })
        
        # 定期保存
        if (idx - start_idx + 1) % save_interval == 0:
            save_results(results, f"hle_avh_results_temp_{start_idx}_{idx+1}.json")
    
    results["accuracy"] = results["correct"] / results["evaluated"] if results["evaluated"] > 0 else 0.0
    
    return results


def save_results(results: Dict[str, Any], filename: str):
    """結果を保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved to {filename}")


def print_summary(results: Dict[str, Any]):
    """結果サマリーを表示"""
    print()
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total questions: {results['total']}")
    print(f"Evaluated: {results['evaluated']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Failed: {results['failed']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print()
    print(f"AVH Cross used: {results['avh_used']}")
    print(f"Verantyx only: {results['verantyx_only']}")
    print()
    
    # カテゴリ別正解率
    category_stats = {}
    for item in results['results']:
        if 'category' in item and 'correct' in item:
            cat = item['category']
            if cat not in category_stats:
                category_stats[cat] = {'total': 0, 'correct': 0}
            category_stats[cat]['total'] += 1
            if item['correct']:
                category_stats[cat]['correct'] += 1
    
    if category_stats:
        print("=" * 80)
        print("Category-wise Accuracy")
        print("=" * 80)
        for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]['correct']):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{cat:40s} {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")


def main():
    """メイン実行"""
    print("=" * 80)
    print("HLE 2500 Evaluation with AVH Integration")
    print("=" * 80)
    print()
    
    # データロード
    print("Loading HLE 2500 questions...")
    questions = load_hle_2500()
    print(f"✅ Loaded {len(questions)} questions")
    print()
    
    # パイプライン初期化
    print("Initializing Verantyx + AVH pipeline...")
    pipeline = VerantyxAvhEnhanced(use_avh=True)
    print("✅ Pipeline ready")
    print()
    
    # バッチサイズ入力
    print("Batch size options:")
    print("  100  - Quick test (4 minutes)")
    print("  500  - Medium test (20 minutes)")
    print("  2500 - Full evaluation (2 hours)")
    batch_size = int(input("Enter batch size (default 100): ") or "100")
    
    # 評価実行
    results = evaluate_hle(
        pipeline=pipeline,
        questions=questions,
        start_idx=0,
        batch_size=batch_size,
        save_interval=50
    )
    
    # 結果保存
    output_file = f"hle_avh_results_{batch_size}.json"
    save_results(results, output_file)
    
    # サマリー表示
    print_summary(results)
    
    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print()
    print(f"Improvement from baseline (3.5%):")
    improvement = (results['accuracy'] - 0.035) * 100
    print(f"  {improvement:+.1f} percentage points")
    print()
    print(f"AVH Cross Simulator usage: {results['avh_used']}/{results['evaluated']} ({results['avh_used']/results['evaluated']*100:.1f}%)")


if __name__ == "__main__":
    main()
