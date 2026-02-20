"""
HLE 100問サンプル検証
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced

try:
    from tqdm import tqdm
except ImportError:
    # tqdmがない場合はシンプルな進捗表示
    def tqdm(iterable, desc=""):
        total = len(list(iterable)) if hasattr(iterable, '__len__') else 0
        print(f"{desc}: {total} items")
        for i, item in enumerate(iterable):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total}")
            yield item

def load_hle_sample(path: str):
    """HLEサンプルをロード"""
    problems = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems

def run_hle_validation(sample_path: str, output_path: str, limit: int = 100):
    """HLE検証を実行"""
    
    # 問題をロード
    problems = load_hle_sample(sample_path)
    if limit:
        problems = problems[:limit]
    
    print(f"Loading {len(problems)} problems...")
    
    # V6初期化
    v6 = VerantyxV6Enhanced()
    
    # 統計
    stats = {
        "total": 0,
        "verified": 0,
        "failed": 0,
        "error": 0,
        "by_type": {},
        "by_domain": {}
    }
    
    results = []
    
    # 各問題を解く
    for i, problem in enumerate(tqdm(problems, desc="Processing")):
        # HLEフォーマットに対応
        question = problem.get("problem_text", problem.get("question", ""))
        answer = problem.get("expected_answer", problem.get("answer", ""))
        problem_type = problem.get("type", "unknown")
        
        stats["total"] += 1
        
        try:
            # 解く（Crystallizer無効）
            result = v6.solve(question, answer, use_crystal=False)
            
            # 統計更新
            status = result["status"]
            if status == "VERIFIED":
                stats["verified"] += 1
            elif status == "FAILED":
                stats["failed"] += 1
            else:
                stats["error"] += 1
            
            # タイプ別統計
            if problem_type not in stats["by_type"]:
                stats["by_type"][problem_type] = {"total": 0, "verified": 0}
            stats["by_type"][problem_type]["total"] += 1
            if status == "VERIFIED":
                stats["by_type"][problem_type]["verified"] += 1
            
            # ドメイン別統計（IRから）
            if "ir" in result:
                domain = result["ir"].get("domain", "unknown")
                if domain not in stats["by_domain"]:
                    stats["by_domain"][domain] = {"total": 0, "verified": 0}
                stats["by_domain"][domain]["total"] += 1
                if status == "VERIFIED":
                    stats["by_domain"][domain]["verified"] += 1
            
            # 結果を保存
            results.append({
                "index": i,
                "question": question,
                "expected": answer,
                "answer": result.get("answer"),
                "status": status,
                "type": problem_type,
                "trace": result.get("trace", [])[-10:]  # 最後の10ステップのみ
            })
            
        except Exception as e:
            stats["error"] += 1
            results.append({
                "index": i,
                "question": question,
                "expected": answer,
                "answer": None,
                "status": "ERROR",
                "type": problem_type,
                "error": str(e)
            })
    
    # 結果を保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "stats": stats,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    # 統計を表示
    print("\n" + "="*80)
    print("HLE Sample Validation Results")
    print("="*80)
    print(f"Total: {stats['total']}")
    print(f"VERIFIED: {stats['verified']} ({stats['verified']/stats['total']*100:.1f}%)")
    print(f"FAILED: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"ERROR: {stats['error']} ({stats['error']/stats['total']*100:.1f}%)")
    print()
    
    # タイプ別統計
    print("By Type:")
    for ptype, pstats in sorted(stats["by_type"].items(), key=lambda x: x[1]["verified"], reverse=True):
        total = pstats["total"]
        verified = pstats["verified"]
        pct = verified / total * 100 if total > 0 else 0
        print(f"  {ptype:20s}: {verified:3d}/{total:3d} ({pct:5.1f}%)")
    
    print()
    
    # ドメイン別統計
    print("By Domain:")
    for domain, dstats in sorted(stats["by_domain"].items(), key=lambda x: x[1]["verified"], reverse=True):
        total = dstats["total"]
        verified = dstats["verified"]
        pct = verified / total * 100 if total > 0 else 0
        print(f"  {domain:20s}: {verified:3d}/{total:3d} ({pct:5.1f}%)")
    
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    return stats

if __name__ == "__main__":
    sample_path = "hle_100_sample.jsonl"
    output_path = "hle_100_results.json"
    
    stats = run_hle_validation(sample_path, output_path, limit=100)
