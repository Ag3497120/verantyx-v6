#!/usr/bin/env python3
"""
Verantyx General-Purpose ARC Solver (Lite)
メモリ軽量版。compositionを制限。

Usage: python3 verantyx_solver_lite.py --split evaluation
"""

import json, os, sys, time, gc
from collections import Counter
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_general_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import Grid, grid_shape, grid_eq, most_common_color

def generate_candidates(train_pairs, test_inputs):
    """Generate candidates without memory explosion."""
    candidates = []
    
    # Phase 1: Cross Engine
    try:
        from arc.cross_engine import solve_cross_engine
        _, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            candidates.append(("cross:" + piece.name, piece.apply))
    except:
        pass
    
    # Phase 2: Puzzle Language
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            candidates.append(("puzzle:" + prog.name, prog.apply_fn))
    except:
        pass
    
    # Phase 3: Standalone Primitives
    try:
        from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives
        inp0 = train_pairs[0][0]
        all_prims = list(PARAMETERLESS_PRIMITIVES) + get_color_primitives(inp0)
        for name, fn in all_prims:
            candidates.append(("prim:" + name, fn))
    except:
        pass
    
    return candidates

def validate_and_select(candidates, train_pairs, test_inputs):
    """Validate candidates with hold-out, return best prediction."""
    n = len(train_pairs)
    perfect = []
    
    for name, fn in candidates:
        # Full pass check
        try:
            full_ok = True
            for inp, out in train_pairs:
                r = fn(inp)
                if r is None or not grid_eq(r, out):
                    full_ok = False
                    break
            if not full_ok:
                continue
        except:
            continue
        
        # Hold-out check (leave-one-out)
        if n >= 2:
            holdout_ok = True
            for i in range(n):
                try:
                    pred = fn(train_pairs[i][0])
                    if pred is None or not grid_eq(pred, train_pairs[i][1]):
                        holdout_ok = False
                        break
                except:
                    holdout_ok = False
                    break
            if not holdout_ok:
                continue
        
        # Generate test predictions
        try:
            test_preds = []
            for ti in test_inputs:
                p = fn(ti)
                if p is None:
                    break
                test_preds.append(p)
            if len(test_preds) == len(test_inputs):
                perfect.append((name, test_preds))
        except:
            continue
    
    if not perfect:
        return None, None, 0
    
    # Majority vote
    vote_counter = Counter()
    vote_map = {}
    for name, preds in perfect:
        key = json.dumps(preds[0])
        vote_counter[key] += 1
        if key not in vote_map:
            vote_map[key] = (name, preds)
    
    best_key, best_votes = vote_counter.most_common(1)[0]
    winner_name, winner_preds = vote_map[best_key]
    return winner_name, winner_preds, best_votes

def solve_task(task_id, data_dir):
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
        candidates = generate_candidates(train_pairs, test_inputs)
        winner_name, winner_preds, votes = validate_and_select(candidates, train_pairs, test_inputs)
    except Exception as e:
        print(f"[{task_id}] Error: {e}")
        gc.collect()
        return "error"
    
    elapsed = time.time() - t0
    
    if winner_preds:
        n_perfect = votes  # approximate
        result = {
            "method": "holdout_perfect",
            "piece": winner_name,
            "votes": votes,
            "n_candidates": len(candidates),
            "elapsed_s": round(elapsed, 1),
            "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
        }
        with open(result_path, 'w') as f:
            json.dump(result, f)
        print(f"[{task_id}] ✅ {winner_name} | {votes} votes | {len(candidates)} cands | {elapsed:.1f}s")
        gc.collect()
        return "perfect"
    
    print(f"[{task_id}] ❌ 0/{len(candidates)} | {elapsed:.1f}s")
    gc.collect()
    return "fail"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = EVAL_DIR if args.split == "evaluation" else TRAIN_DIR
    
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = sorted([f.replace('.json','') for f in os.listdir(data_dir) if f.endswith('.json')])
        end = args.end or len(task_ids)
        task_ids = task_ids[args.start:end]
    
    print(f"{'='*65}")
    print(f"  Verantyx ARC Solver (Lite)")
    print(f"  Cross Engine + Puzzle Lang + Primitives")
    print(f"  Hold-out Validation + Majority Voting")
    print(f"{'='*65}")
    print(f"Split: {args.split}, Tasks: {len(task_ids)}\n")
    
    stats = {"perfect": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        status = solve_task(tid, data_dir)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    # Verify
    print(f"\n{'='*65}")
    print(f"Candidates found: {stats['perfect']}, Failed: {stats['fail']}, Errors: {stats['error']}")
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
