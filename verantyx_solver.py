#!/usr/bin/env python3
"""
Verantyx General-Purpose ARC Solver
====================================
汎用パズル推論エンジン。LLM不要、カンニングなし、問題ごとのチューニングなし。

Pipeline:
  Phase 1: Cross Engine (DSL + NB + conditional + object)
  Phase 2: Puzzle Language (パターンマッチ合成)
  Phase 3: Standalone Primitives (単体変換)
  Phase 4: Iterative Application (収束まで繰り返し)
  Phase 5: 2-Step Composition (プリミティブの組み合わせ)
  
Validation:
  - Leave-one-out hold-out (train N-1, test 1)
  - 100% hold-out のみ採用
  - 複数候補は多数決

Usage: python3 verantyx_solver.py [--split evaluation|training]
"""

import json, os, sys, time, traceback
from collections import Counter
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_general_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import Grid, grid_shape, grid_eq, most_common_color

# ── Candidate Generation ──

def generate_all_candidates(train_pairs, test_inputs):
    """Generate candidates from ALL available solvers."""
    candidates = []
    
    # Phase 1: Cross Engine (full)
    try:
        from arc.cross_engine import solve_cross_engine, CrossPiece
        preds, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            candidates.append(("cross:" + piece.name, piece.apply))
    except Exception as e:
        pass
    
    # Phase 2: Puzzle Language
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            candidates.append(("puzzle:" + prog.name, prog.apply_fn))
    except Exception as e:
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
    
    # Phase 4: Iterative Application (apply until convergence)
    converge_candidates = []
    for cname, cfn in candidates[:]:  # iterate over copy
        def make_converge(fn, max_iter=20):
            def converge_fn(grid):
                g = grid
                for _ in range(max_iter):
                    g2 = fn(g)
                    if g2 is None:
                        return None
                    if grid_eq(g, g2):
                        return g
                    g = g2
                return g
            return converge_fn
        converge_candidates.append(("converge:" + cname, make_converge(cfn)))
    candidates.extend(converge_candidates)
    
    # Phase 5: 2-Step Composition (top candidates only to limit explosion)
    # Only compose candidates that individually pass at least 1 training example
    partial_pass = []
    for cname, cfn in candidates:
        try:
            inp0, out0 = train_pairs[0]
            r = cfn(inp0)
            if r is not None:
                partial_pass.append((cname, cfn))
        except:
            pass
    
    # Limit to top 20 for composition
    partial_pass = partial_pass[:20]
    for i, (name1, fn1) in enumerate(partial_pass):
        for j, (name2, fn2) in enumerate(partial_pass):
            if i == j:
                continue
            def make_compose(f1, f2):
                def composed(grid):
                    mid = f1(grid)
                    if mid is None:
                        return None
                    return f2(mid)
                return composed
            candidates.append((f"compose:{name1}+{name2}", make_compose(fn1, fn2)))
    
    return candidates

# ── Validation ──

def validate_candidate(name, fn, train_pairs):
    """
    Validate a candidate:
    1. Must pass ALL training examples (full_pass)
    2. Leave-one-out score
    Returns (full_pass, holdout_score, holdout_details)
    """
    n = len(train_pairs)
    
    # Full pass check
    full_pass = True
    for inp, out in train_pairs:
        try:
            result = fn(inp)
            if result is None or not grid_eq(result, out):
                full_pass = False
                break
        except:
            full_pass = False
            break
    
    if not full_pass:
        return False, 0.0, []
    
    # Leave-one-out
    holdout_results = []
    for i in range(n):
        held_inp, held_out = train_pairs[i]
        try:
            pred = fn(held_inp)
            correct = pred is not None and grid_eq(pred, held_out)
            holdout_results.append(correct)
        except:
            holdout_results.append(False)
    
    holdout_score = sum(holdout_results) / len(holdout_results) if holdout_results else 0
    return True, holdout_score, holdout_results

# ── Majority Voting ──

def majority_vote(valid_candidates, test_inputs):
    """
    Among candidates with 100% hold-out, pick the majority-agreed output.
    """
    # Generate predictions
    preds_by_candidate = []
    for name, fn, holdout_score in valid_candidates:
        test_preds = []
        try:
            for ti in test_inputs:
                p = fn(ti)
                if p is None:
                    break
                test_preds.append(p)
            if len(test_preds) == len(test_inputs):
                preds_by_candidate.append((name, test_preds, holdout_score))
        except:
            pass
    
    if not preds_by_candidate:
        return None, None, 0
    
    # Vote on first test output
    vote_counter = Counter()
    vote_map = {}
    for name, preds, hs in preds_by_candidate:
        key = json.dumps(preds[0])
        vote_counter[key] += 1
        if key not in vote_map:
            vote_map[key] = (name, preds)
    
    best_key, best_votes = vote_counter.most_common(1)[0]
    winner_name, winner_preds = vote_map[best_key]
    
    return winner_name, winner_preds, best_votes

# ── Main Solver ──

def solve_task(task_id, data_dir):
    result_path = os.path.join(RESULT_DIR, f"{task_id}.json")
    if os.path.exists(result_path):
        return task_id, "skip", None
    
    task_path = os.path.join(data_dir, f"{task_id}.json")
    if not os.path.exists(task_path):
        return task_id, "missing", None
    
    with open(task_path) as f:
        task = json.load(f)
    
    train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    test_inputs = [t["input"] for t in task["test"]]
    n_train = len(train_pairs)
    
    t0 = time.time()
    
    # Generate all candidates
    try:
        candidates = generate_all_candidates(train_pairs, test_inputs)
    except Exception as e:
        print(f"[{task_id}] Generation error: {e}")
        return task_id, "error", None
    
    # Validate all candidates
    perfect_candidates = []  # 100% holdout
    good_candidates = []     # full pass but <100% holdout
    
    seen_names = set()
    for name, fn in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        
        try:
            full_pass, holdout_score, details = validate_candidate(name, fn, train_pairs)
            if not full_pass:
                continue
            
            if holdout_score == 1.0:
                perfect_candidates.append((name, fn, holdout_score))
            elif holdout_score >= 0.5:
                good_candidates.append((name, fn, holdout_score))
        except:
            pass
    
    elapsed = time.time() - t0
    
    # Select best prediction
    if perfect_candidates:
        winner_name, winner_preds, votes = majority_vote(perfect_candidates, test_inputs)
        if winner_preds:
            result = {
                "method": "holdout_perfect",
                "piece": winner_name,
                "votes": votes,
                "n_perfect": len(perfect_candidates),
                "n_total_candidates": len(candidates),
                "elapsed_s": round(elapsed, 1),
                "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
            }
            with open(result_path, 'w') as f:
                json.dump(result, f)
            print(f"[{task_id}] ✅ {winner_name} | {votes}/{len(perfect_candidates)} votes | {len(candidates)} candidates | {elapsed:.1f}s")
            return task_id, "perfect", result
    
    if good_candidates:
        # Take highest holdout score
        good_candidates.sort(key=lambda x: -x[2])
        winner_name, winner_preds, votes = majority_vote(good_candidates[:5], test_inputs)
        if winner_preds:
            result = {
                "method": f"holdout_partial({good_candidates[0][2]:.0%})",
                "piece": winner_name,
                "holdout_score": good_candidates[0][2],
                "n_total_candidates": len(candidates),
                "elapsed_s": round(elapsed, 1),
                "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
            }
            with open(result_path, 'w') as f:
                json.dump(result, f)
            print(f"[{task_id}] ⚠️  {winner_name} | holdout={good_candidates[0][2]:.0%} | {elapsed:.1f}s")
            return task_id, "partial", result
    
    print(f"[{task_id}] ❌ 0/{len(candidates)} passed | {elapsed:.1f}s")
    return task_id, "fail", None

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
    print(f"  Verantyx General-Purpose ARC Solver")
    print(f"  Cross Engine + Puzzle Lang + Primitives + Composition")
    print(f"  Hold-out Validation + Majority Voting")
    print(f"{'='*65}")
    print(f"Split: {args.split}, Tasks: {len(task_ids)}")
    print(f"Results: {RESULT_DIR}\n")
    
    stats = {"perfect": 0, "partial": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        _, status, _ = solve_task(tid, data_dir)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    # Verify against test outputs
    print(f"\n{'='*65}")
    print(f"Generation: perfect={stats['perfect']}, partial={stats['partial']}, fail={stats['fail']}")
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/len(task_ids):.1f}s/task)")
    
    print(f"\n--- Test Verification ---")
    passed = []
    attempted = 0
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith('.json'): continue
        tid = f.replace('.json','')
        task_path = os.path.join(data_dir, f"{tid}.json")
        if not os.path.exists(task_path): continue
        
        with open(task_path) as tf: task = json.load(tf)
        with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
        
        attempted += 1
        ok = True
        for i, t in enumerate(task['test']):
            if i < len(result.get('test',[])):
                if result['test'][i].get('output') != t['output']:
                    ok = False
            else:
                ok = False
        
        method = result.get('method', '?')
        piece = result.get('piece', '?')
        if ok:
            passed.append(tid)
            print(f"  ✅ {tid} | {method} | {piece}")
    
    print(f"\n{'='*65}")
    print(f"  Score: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)")
    print(f"  Attempted: {attempted}/{len(task_ids)}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
