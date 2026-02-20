#!/usr/bin/env python3
"""
HLE 2500問完全評価 - Phase 5G完了後
"""
import sys
import json
import time
from collections import defaultdict
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


def evaluate_full():
    """HLE 2500問完全評価"""
    pipeline = VerantyxV6Enhanced()
    
    # HLEデータをロード
    with open('hle_2500_eval.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    total = len(questions)
    correct = 0
    failed = 0
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    print("=" * 80)
    print(f"HLE 2500 Full Evaluation (Phase 5G Complete)")
    print("=" * 80)
    print(f"Total questions: {total}")
    print()
    
    start_time = time.time()
    
    for i, q in enumerate(questions):
        question = q['question']
        expected = q['answer']
        category = q.get('category', 'Unknown')
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i + 1}/{total} ({correct} correct, {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min)")
        
        try:
            result = pipeline.solve(question)
            answer = result.get('answer')
            
            category_stats[category]["total"] += 1
            
            # flexible_matchで判定
            if answer is not None and expected is not None:
                if flexible_match(answer, expected, tolerance=1e-4):
                    correct += 1
                    category_stats[category]["correct"] += 1
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Error on Q{i+1} ({category}): {str(e)[:100]}")
    
    total_time = time.time() - start_time
    accuracy = (correct / total) * 100
    
    print()
    print("=" * 80)
    print("Overall Results:")
    print("=" * 80)
    print(f"Correct: {correct}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg time per question: {total_time/total:.2f} seconds")
    print()
    
    print("=" * 80)
    print("Category Breakdown:")
    print("=" * 80)
    print(f"{'Category':<30} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
    print("-" * 80)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        cat_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"{category:<30} {stats['correct']:>10} {stats['total']:>10} {cat_acc:>9.1f}%")
    
    print()
    print("=" * 80)
    print("Comparison:")
    print("=" * 80)
    print(f"Baseline (before Phase 5G): 3.5%")
    print(f"Current: {accuracy:.2f}%")
    print(f"Improvement: {accuracy - 3.5:+.2f} points")
    print()
    
    if accuracy >= 10.0:
        print("🎉 目標10%達成！")
    elif accuracy >= 7.0:
        print("📈 大幅改善（2倍以上）")
    elif accuracy >= 5.0:
        print("📊 顕著な改善")
    else:
        print("⚠️  目標未達、さらなる改善が必要")
    
    print("=" * 80)
    
    # 結果を保存
    results = {
        "total": total,
        "correct": correct,
        "failed": failed,
        "accuracy": accuracy,
        "time": total_time,
        "category_stats": dict(category_stats)
    }
    
    with open('hle_2500_phase5g_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to hle_2500_phase5g_results.json")


if __name__ == "__main__":
    evaluate_full()
