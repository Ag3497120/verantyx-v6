#!/usr/bin/env python3
"""
Verantyx ARC Solver v2
- 真のLeave-one-out: N-1例で再学習→残り1例を予測
- 多数決 + 簡潔性優先
"""

import json, os, sys, time, gc
from collections import Counter
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_v2_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import grid_eq

def get_candidates(train_pairs, test_inputs):
    """Generate named candidates."""
    candidates = []
    
    # Cross Engine
    try:
        from arc.cross_engine import solve_cross_engine
        _, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            candidates.append(("cross:" + piece.name, piece.apply))
    except:
        pass
    
    # Puzzle Language
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            candidates.append(("puzzle:" + prog.name, prog.apply_fn))
    except:
        pass
    
    # Primitives
    try:
        from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives
        inp0 = train_pairs[0][0]
        all_prims = list(PARAMETERLESS_PRIMITIVES) + get_color_primitives(inp0)
        for name, fn in all_prims:
            candidates.append(("prim:" + name, fn))
    except:
        pass
    
    return candidates

def true_holdout(train_pairs, test_inputs):
    """
    True leave-one-out cross-validation.
    For each held-out example:
      1. Re-generate candidates from N-1 examples
      2. Find candidates that pass all N-1 examples
      3. Predict held-out example
      4. Check if prediction is correct
    
    Only candidates that survive ALL hold-out folds are kept.
    """
    n = len(train_pairs)
    
    # Step 1: Get full-training candidates (must pass ALL examples)
    full_candidates = get_candidates(train_pairs, test_inputs)
    
    # Filter to those passing all training examples
    full_pass = []
    for name, fn in full_candidates:
        try:
            ok = True
            for inp, out in train_pairs:
                r = fn(inp)
                if r is None or not grid_eq(r, out):
                    ok = False
                    break
            if ok:
                full_pass.append((name, fn))
        except:
            pass
    
    if not full_pass or n < 2:
        return full_pass  # Can't do hold-out with < 2 examples
    
    # Step 2: For each fold, re-learn and check
    # Key insight: a candidate from full training might not appear
    # when trained on N-1. If it does AND predicts the held-out
    # correctly, it's truly generalizing.
    
    holdout_survivors = []
    
    for name, fn in full_pass:
        survived_all = True
        
        for i in range(n):
            # Hold out example i
            held_inp, held_out = train_pairs[i]
            reduced_pairs = train_pairs[:i] + train_pairs[i+1:]
            
            # Re-generate candidates from reduced set
            try:
                reduced_candidates = get_candidates(reduced_pairs, [held_inp])
            except:
                survived_all = False
                break
            
            # Find the SAME named candidate in reduced set
            found_match = False
            for rname, rfn in reduced_candidates:
                if rname != name:
                    continue
                
                # Verify it passes all reduced training
                try:
                    rok = True
                    for rinp, rout in reduced_pairs:
                        rr = rfn(rinp)
                        if rr is None or not grid_eq(rr, rout):
                            rok = False
                            break
                    if not rok:
                        continue
                except:
                    continue
                
                # Predict held-out
                try:
                    pred = rfn(held_inp)
                    if pred is not None and grid_eq(pred, held_out):
                        found_match = True
                        break
                except:
                    pass
            
            if not found_match:
                survived_all = False
                break
        
        if survived_all:
            holdout_survivors.append((name, fn))
    
    return holdout_survivors

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
        survivors = true_holdout(train_pairs, test_inputs)
    except Exception as e:
        print(f"[{task_id}] Error: {e}")
        gc.collect()
        return "error"
    
    elapsed = time.time() - t0
    
    if not survivors:
        # Fallback: try full-pass candidates without holdout
        # but mark as low confidence
        try:
            full_candidates = get_candidates(train_pairs, test_inputs)
            full_pass = []
            for name, fn in full_candidates:
                try:
                    ok = all(fn(inp) is not None and grid_eq(fn(inp), out) for inp, out in train_pairs)
                    if ok:
                        full_pass.append((name, fn))
                except:
                    pass
            
            if full_pass:
                # Use majority vote among full-pass (no holdout guarantee)
                vote_counter = Counter()
                vote_map = {}
                for name, fn in full_pass:
                    try:
                        preds = [fn(ti) for ti in test_inputs]
                        if all(p is not None for p in preds):
                            key = json.dumps(preds[0])
                            vote_counter[key] += 1
                            if key not in vote_map:
                                vote_map[key] = (name, preds)
                    except:
                        pass
                
                if vote_counter:
                    best_key, best_votes = vote_counter.most_common(1)[0]
                    total_voters = sum(vote_counter.values())
                    
                    # Only use if strong consensus (>50% agree)
                    if best_votes > total_voters * 0.5 and best_votes >= 2:
                        winner_name, winner_preds = vote_map[best_key]
                        result = {
                            "method": f"consensus({best_votes}/{total_voters})",
                            "piece": winner_name,
                            "votes": best_votes,
                            "total_voters": total_voters,
                            "elapsed_s": round(elapsed, 1),
                            "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
                        }
                        with open(result_path, 'w') as f:
                            json.dump(result, f)
                        print(f"[{task_id}] ⚠️  {winner_name} | consensus {best_votes}/{total_voters} | {elapsed:.1f}s")
                        gc.collect()
                        return "consensus"
        except:
            pass
        
        print(f"[{task_id}] ❌ | {elapsed:.1f}s")
        gc.collect()
        return "fail"
    
    # Majority vote among holdout survivors
    vote_counter = Counter()
    vote_map = {}
    for name, fn in survivors:
        try:
            preds = [fn(ti) for ti in test_inputs]
            if all(p is not None for p in preds):
                key = json.dumps(preds[0])
                vote_counter[key] += 1
                if key not in vote_map:
                    vote_map[key] = (name, preds)
        except:
            pass
    
    if not vote_counter:
        print(f"[{task_id}] ❌ survivors couldn't predict | {elapsed:.1f}s")
        gc.collect()
        return "fail"
    
    best_key, best_votes = vote_counter.most_common(1)[0]
    winner_name, winner_preds = vote_map[best_key]
    
    result = {
        "method": "true_holdout",
        "piece": winner_name,
        "votes": best_votes,
        "n_survivors": len(survivors),
        "elapsed_s": round(elapsed, 1),
        "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
    }
    with open(result_path, 'w') as f:
        json.dump(result, f)
    
    print(f"[{task_id}] ✅ {winner_name} | {best_votes}/{len(survivors)} votes | holdout | {elapsed:.1f}s")
    gc.collect()
    return "holdout"

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
    print(f"  Verantyx ARC Solver v2")
    print(f"  True Leave-One-Out + Consensus Voting")
    print(f"{'='*65}")
    print(f"Split: {args.split}, Tasks: {len(task_ids)}\n")
    
    stats = {"holdout": 0, "consensus": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        status = solve_task(tid, data_dir)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    print(f"\n{'='*65}")
    print(f"Holdout pass: {stats['holdout']}, Consensus: {stats['consensus']}, Fail: {stats['fail']}")
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
        method = result.get('method','?')
        if ok:
            passed.append(tid)
            print(f"  ✅ {tid} | {result.get('piece','?')} | {method}")
    
    print(f"\n  Score: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)")
    print(f"  Attempted: {attempted}/{len(task_ids)}")

if __name__ == "__main__":
    main()
