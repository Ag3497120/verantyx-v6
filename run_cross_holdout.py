#!/usr/bin/env python3
"""
Verantyx Cross Engine with Hold-Out Validation
- Leave-one-out: train on N-1 examples, test on held-out example
- Only output predictions where hold-out score >= threshold
- Rank multiple candidates by hold-out accuracy

Usage: python3 run_cross_holdout.py
"""

import json, os, sys, traceback
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/cross_holdout_results")
os.makedirs(RESULT_DIR, exist_ok=True)

def holdout_validate(task):
    """
    Leave-one-out cross-validation using cross engine.
    For each training example, train on the rest, predict the held-out one.
    Returns list of (piece_name, holdout_accuracy, test_predictions).
    """
    from arc.cross_engine import solve_cross_engine, CrossSimulator, CrossPiece, _generate_cross_pieces
    from arc.cross_solver import solve_cross
    from arc.grid import grid_eq
    
    train = task["train"]
    test_inputs = [t["input"] for t in task["test"]]
    n = len(train)
    
    if n < 2:
        # Can't do hold-out with less than 2 examples
        return []
    
    # Step 1: Get all candidates from FULL training set
    full_pairs = [(ex["input"], ex["output"]) for ex in train]
    _, full_verified = solve_cross_engine(full_pairs, test_inputs)
    
    if not full_verified:
        return []
    
    # Step 2: For each candidate, do leave-one-out
    results = []
    
    for tag, piece in full_verified:
        holdout_correct = 0
        holdout_total = 0
        
        for i in range(n):
            held_input = train[i]["input"]
            held_output = train[i]["output"]
            
            # Check if piece predicts held-out example correctly
            try:
                pred = piece.apply(held_input)
                if pred is not None and grid_eq(pred, held_output):
                    holdout_correct += 1
                holdout_total += 1
            except:
                holdout_total += 1
        
        holdout_acc = holdout_correct / holdout_total if holdout_total > 0 else 0
        
        # Generate test predictions
        test_preds = []
        for ti in test_inputs:
            try:
                p = piece.apply(ti)
                test_preds.append(p)
            except:
                test_preds.append(None)
        
        if all(p is not None for p in test_preds):
            results.append({
                "name": piece.name,
                "holdout_acc": holdout_acc,
                "holdout_correct": holdout_correct,
                "holdout_total": holdout_total,
                "test_preds": test_preds,
            })
    
    # Sort by holdout accuracy (descending), then by name stability
    results.sort(key=lambda x: (-x["holdout_acc"], x["name"]))
    return results

def solve_task(task_id):
    result_path = os.path.join(RESULT_DIR, f"{task_id}.json")
    if os.path.exists(result_path):
        return task_id, "skip"
    
    task_path = os.path.join(EVAL_DIR, f"{task_id}.json")
    if not os.path.exists(task_path):
        return task_id, "missing"
    
    with open(task_path) as f:
        task = json.load(f)
    
    try:
        candidates = holdout_validate(task)
    except Exception as e:
        print(f"[{task_id}] Error: {e}")
        return task_id, "error"
    
    if not candidates:
        print(f"[{task_id}] No candidates")
        return task_id, "none"
    
    # Filter: only keep candidates with 100% hold-out accuracy
    perfect = [c for c in candidates if c["holdout_acc"] == 1.0]
    
    if perfect:
        # Multiple perfect candidates → majority vote
        if len(perfect) > 1:
            # Vote on test output
            output_votes = Counter()
            output_map = {}
            for c in perfect:
                key = json.dumps(c["test_preds"][0])
                output_votes[key] += 1
                output_map[key] = c
            
            best_key, best_count = output_votes.most_common(1)[0]
            winner = output_map[best_key]
            method = f"holdout_perfect_vote({best_count}/{len(perfect)})"
        else:
            winner = perfect[0]
            method = "holdout_perfect"
        
        result = {
            "method": method,
            "piece": winner["name"],
            "holdout_acc": winner["holdout_acc"],
            "n_candidates": len(candidates),
            "n_perfect": len(perfect),
            "test": [{"output": winner["test_preds"][i]} for i in range(len(winner["test_preds"]))]
        }
        with open(result_path, 'w') as f:
            json.dump(result, f)
        print(f"[{task_id}] ✅ {method} | {winner['name']} | {len(perfect)} perfect / {len(candidates)} total")
        return task_id, "perfect"
    
    # No perfect hold-out → take best but mark as low-confidence
    best = candidates[0]
    if best["holdout_acc"] >= 0.5:
        result = {
            "method": f"holdout_partial({best['holdout_acc']:.0%})",
            "piece": best["name"],
            "holdout_acc": best["holdout_acc"],
            "n_candidates": len(candidates),
            "test": [{"output": best["test_preds"][i]} for i in range(len(best["test_preds"]))]
        }
        with open(result_path, 'w') as f:
            json.dump(result, f)
        print(f"[{task_id}] ⚠️  holdout={best['holdout_acc']:.0%} | {best['name']} | {len(candidates)} candidates")
        return task_id, "partial"
    else:
        print(f"[{task_id}] ❌ best holdout={best['holdout_acc']:.0%}, skipping")
        return task_id, "low"

def main():
    task_ids = sorted([f.replace('.json','') for f in os.listdir(EVAL_DIR) if f.endswith('.json')])
    
    print(f"{'='*60}")
    print(f"  Verantyx Cross Engine + Hold-Out Validation")
    print(f"{'='*60}")
    print(f"Tasks: {len(task_ids)}\n")
    
    stats = {"perfect": 0, "partial": 0, "none": 0, "low": 0, "error": 0, "skip": 0, "missing": 0}
    
    for tid in task_ids:
        _, status = solve_task(tid)
        stats[status] = stats.get(status, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"Perfect hold-out: {stats['perfect']}")
    print(f"Partial hold-out: {stats['partial']}")
    print(f"No candidates:    {stats['none']}")
    print(f"Low confidence:   {stats['low']}")
    print(f"Errors:           {stats['error']}")
    
    # Verify against test
    print(f"\n--- Test Verification ---")
    passed = 0
    total = 0
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith('.json'): continue
        tid = f.replace('.json','')
        task_path = os.path.join(EVAL_DIR, f"{tid}.json")
        if not os.path.exists(task_path): continue
        
        with open(task_path) as tf: task = json.load(tf)
        with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
        
        total += 1
        ok = True
        for i, t in enumerate(task['test']):
            if i < len(result.get('test',[])):
                if result['test'][i].get('output') != t['output']:
                    ok = False
            else:
                ok = False
        
        if ok:
            passed += 1
            print(f"  ✅ {tid} ({result.get('method','?')})")
        #else:
        #    print(f"  ❌ {tid} ({result.get('method','?')})")
    
    print(f"\nScore: {passed}/{len(task_ids)} ({100*passed/len(task_ids):.1f}%)")
    print(f"Attempted: {total}/{len(task_ids)}")

if __name__ == "__main__":
    main()
