#!/usr/bin/env python3
"""
Verantyx World Model Solver
============================
crossシミュレータ + 世界モデル + パズル推論統合版。

Pipeline:
  1. World Model: 物理法則コマンドの組み合わせ探索（depth 1-3 + converge）
  2. Cross Engine: 既存94モジュール
  3. Puzzle Language: パターンマッチ
  4. 多数決で最終選択

Usage: python3 verantyx_wm_solver.py --split evaluation
"""

import json, os, sys, time, gc
from collections import Counter
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_wm_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import grid_eq
from arc.world_model import WorldModel

wm = WorldModel()

def solve_task(task_id, data_dir, timeout=120):
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
    all_solutions = []
    
    # ── Phase 1: World Model ──
    try:
        wm_solutions = wm.solve(train_pairs, test_inputs, max_depth=3, timeout_s=timeout*0.6)
        all_solutions.extend(wm_solutions)
    except Exception as e:
        pass
    
    # ── Phase 2: Cross Engine ──
    try:
        from arc.cross_engine import solve_cross_engine
        _, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            name = getattr(piece, 'name', type(piece).__name__)
            try:
                preds = [piece.apply(ti) for ti in test_inputs]
                if all(p is not None for p in preds):
                    all_solutions.append((f"cross:{name}", preds, 0.95))
            except:
                pass
    except:
        pass
    
    # ── Phase 3: Puzzle Language ──
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            try:
                ok = all(grid_eq(prog.apply_fn(i), o) for i, o in train_pairs)
                if ok:
                    preds = [prog.apply_fn(ti) for ti in test_inputs]
                    if all(p is not None for p in preds):
                        all_solutions.append((f"puzzle:{prog.name}", preds, 0.9))
            except:
                pass
    except:
        pass
    
    elapsed = time.time() - t0
    
    if not all_solutions:
        print(f"[{task_id}] ❌ | {elapsed:.1f}s")
        gc.collect()
        return "fail"
    
    # ── 多数決 ──
    vote_counter = Counter()
    vote_map = {}
    for name, preds, conf in all_solutions:
        key = json.dumps(preds[0])
        vote_counter[key] += conf  # confidence-weighted
        if key not in vote_map or conf > vote_map[key][2]:
            vote_map[key] = (name, preds, conf)
    
    best_key, best_score = vote_counter.most_common(1)[0]
    winner_name, winner_preds, winner_conf = vote_map[best_key]
    
    # Count unique outputs
    n_unique = len(vote_counter)
    n_solutions = len(all_solutions)
    
    result = {
        "method": "world_model",
        "piece": winner_name,
        "confidence": round(winner_conf, 2),
        "n_solutions": n_solutions,
        "n_unique": n_unique,
        "weighted_score": round(best_score, 2),
        "elapsed_s": round(elapsed, 1),
        "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
    }
    with open(result_path, 'w') as f:
        json.dump(result, f)
    
    print(f"[{task_id}] ✅ {winner_name} | {n_solutions} solutions ({n_unique} unique) | {elapsed:.1f}s")
    gc.collect()
    return "solved"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    
    data_dir = EVAL_DIR if args.split == "evaluation" else TRAIN_DIR
    
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = sorted([f.replace('.json','') for f in os.listdir(data_dir) if f.endswith('.json')])
        end = args.end or len(task_ids)
        task_ids = task_ids[args.start:end]
    
    print(f"{'='*65}")
    print(f"  Verantyx World Model Solver")
    print(f"  Physics Commands × Program Search × Simulation Verify")
    print(f"{'='*65}")
    print(f"Split: {args.split}, Tasks: {len(task_ids)}, Timeout: {args.timeout}s/task\n")
    
    stats = {"solved": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        status = solve_task(tid, data_dir, args.timeout)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    print(f"\n{'='*65}")
    print(f"Solved: {stats['solved']}, Failed: {stats['fail']}")
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/max(len(task_ids),1):.1f}s/task)\n")
    
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
            print(f"  ✅ {tid} | {result.get('piece','?')}")
    
    print(f"\n  Score: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)")
    print(f"  Attempted: {attempted}/{len(task_ids)}")

if __name__ == "__main__":
    main()
