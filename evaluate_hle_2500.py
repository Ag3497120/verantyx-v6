"""
HLE 2500 Full Evaluation Script

Verantyx V6で2500問全問を評価
"""
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


def load_hle_2500(file_path: str = "hle_2500_eval.jsonl") -> List[Dict[str, Any]]:
    """HLE 2500データをロード"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def evaluate_batch(
    pipeline: VerantyxV6Enhanced,
    questions: List[Dict[str, Any]],
    start_idx: int = 0,
    batch_size: int = 100,
    save_interval: int = 50
) -> Dict[str, Any]:
    """
    バッチ評価
    
    Args:
        pipeline: Verantyx V6パイプライン
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
        "start_time": datetime.now().isoformat(),
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
            
            answer = result.get('answer')
            status = result.get('status')
            
            # 正解判定（flexible_match使用）
            is_correct = False
            if answer is not None and expected_answer is not None:
                is_correct = flexible_match(answer, expected_answer, tolerance=1e-4)
            
            # 結果記録
            item_result = {
                "id": q_id,
                "category": category,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "expected": expected_answer,
                "answer": answer,
                "correct": is_correct,
                "status": status,
                "elapsed": round(elapsed, 2)
            }
            
            results["results"].append(item_result)
            results["evaluated"] += 1
            
            if is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1
            
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
            save_results(results, f"hle_2500_results_temp_{start_idx}_{idx+1}.json")
    
    results["end_time"] = datetime.now().isoformat()
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
    print("HLE 2500 Full Evaluation")
    print("=" * 80)
    print()
    
    # データロード
    print("Loading HLE 2500 questions...")
    questions = load_hle_2500()
    print(f"✅ Loaded {len(questions)} questions")
    print()
    
    # パイプライン初期化
    print("Initializing Verantyx V6...")
    pipeline = VerantyxV6Enhanced(use_beam_search=False, use_simulation=False)
    print("✅ Pipeline ready")
    print()
    
    # 評価実行（バッチ処理）
    # 最初の100問でテスト
    batch_size = int(input("Batch size (default 100, enter 2500 for full): ") or "100")
    
    results = evaluate_batch(
        pipeline=pipeline,
        questions=questions,
        start_idx=0,
        batch_size=batch_size,
        save_interval=50
    )
    
    # 結果保存
    output_file = f"hle_2500_results_{batch_size}.json"
    save_results(results, output_file)
    
    # サマリー表示
    print_summary(results)
    
    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
