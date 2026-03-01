#!/usr/bin/env python3
"""
Verantyx Cross World Model Solver
===================================
Cross Engine（全94モジュール）+ 世界法則コマンド（合成・収束）

Usage: python3 verantyx_cwm_solver.py --split evaluation
"""

import json, os, sys, time, gc
from collections import Counter
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_cwm_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import grid_eq
from arc.cross_world_model import CrossWorldModel

cwm = CrossWorldModel()

def solve_task(task_id, data_dir, timeout=90):
    result_path = os.path.join(RESULT_DIR, f"{task_id}.json")
    if os.path.exists(result_path):
        return "skip"
    
    task_path = os.path.join(data_dir, f"{task_id}.json")
    if not os.path.exists(task_path):
        return "missing"
    
    with open(task_path) as f:
        task = json.load(f)
    
    train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    test_inputs = [t["input"] for t in task["test"]]
    
    t0 = time.time()
    
    try:
        solutions = cwm.solve(train_pairs, test_inputs, timeout_s=timeout)
    except Exception as e:
        print(f"[{task_id}] Error: {e}", flush=True)
        gc.collect()
        return "error"
    
    elapsed = time.time() - t0
    
    if not solutions:
        print(f"[{task_id}] ❌ | {elapsed:.1f}s", flush=True)
        gc.collect()
        return "fail"
    
    # 多数決（confidence加重）
    vote = Counter()
    vote_map = {}
    for sol in solutions:
        key = json.dumps(sol.preds[0])
        vote[key] += sol.confidence
        if key not in vote_map or sol.confidence > vote_map[key].confidence:
            vote_map[key] = sol
    
    best_key, best_score = vote.most_common(1)[0]
    winner = vote_map[best_key]
    
    result = {
        "method": winner.source,
        "piece": winner.name,
        "confidence": round(winner.confidence, 2),
        "n_solutions": len(solutions),
        "n_unique": len(vote),
        "elapsed_s": round(elapsed, 1),
        "test": [{"output": winner.preds[i]} for i in range(len(winner.preds))]
    }
    with open(result_path, 'w') as f:
        json.dump(result, f)
    
    src = winner.source
    print(f"[{task_id}] ✅ {winner.name} | {src} | {len(solutions)} sol | {elapsed:.1f}s", flush=True)
    gc.collect()
    return "solved"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    
    data_dir = EVAL_DIR if args.split == "evaluation" else TRAIN_DIR
    
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = sorted([f.replace('.json','') for f in os.listdir(data_dir) if f.endswith('.json')])
        end = args.end or len(task_ids)
        task_ids = task_ids[args.start:end]
    
    print(f"{'='*65}", flush=True)
    print(f"  Verantyx Cross World Model Solver", flush=True)
    print(f"  Cross Engine (94 modules) + World Laws + Composition", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"Split: {args.split}, Tasks: {len(task_ids)}, Timeout: {args.timeout}s\n", flush=True)
    
    stats = {"solved": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        status = solve_task(tid, data_dir, args.timeout)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    print(f"\n{'='*65}", flush=True)
    print(f"Solved: {stats['solved']}, Failed: {stats['fail']}, Errors: {stats['error']}", flush=True)
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/max(len(task_ids),1):.1f}s/task)\n", flush=True)
    
    # Verify
    passed = []
    attempted = 0
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith('.json'): continue
        tid = f.replace('.json','')
        tp = os.path.join(data_dir, f"{tid}.json")
        if not os.path.exists(tp): continue
        with open(tp) as tf: task = json.load(tf)
        with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
        attempted += 1
        ok = all(
            i < len(result.get('test',[])) and result['test'][i].get('output') == t['output']
            for i, t in enumerate(task['test'])
        )
        if ok:
            passed.append(tid)
            print(f"  ✅ {tid} | {result.get('piece','?')} | {result.get('method','?')}", flush=True)
    
    print(f"\n  Score: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)", flush=True)
    print(f"  Attempted: {attempted}/{len(task_ids)}", flush=True)

if __name__ == "__main__":
    main()
