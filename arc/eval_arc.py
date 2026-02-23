"""
arc/eval_arc.py — Evaluate Verantyx on ARC-AGI-2

Usage:
  python3 -m arc.eval_arc [--data-dir /path/to/ARC-AGI-2/data] [--split training|evaluation]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from arc.solver import load_task, solve_task
from arc.grid import grid_eq, grid_to_str


def evaluate(data_dir: str, split: str = "training", limit: int = 0, verbose: bool = False):
    task_dir = os.path.join(data_dir, split)
    if not os.path.isdir(task_dir):
        print(f"Error: {task_dir} not found")
        sys.exit(1)
    
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])
    if limit > 0:
        task_files = task_files[:limit]
    
    total = len(task_files)
    correct = 0
    attempted = 0
    methods = {}
    errors = 0
    
    start = time.time()
    
    for i, tf in enumerate(task_files):
        path = os.path.join(task_dir, tf)
        try:
            task = load_task(path)
            result = solve_task(task)
            
            # Check correctness (pass@2: correct if ANY attempt matches)
            task_correct = True
            task_attempted = False
            
            for ti, test_preds in enumerate(result.predictions):
                if not test_preds:
                    task_correct = False
                    continue
                task_attempted = True
                
                # Get ground truth
                with open(path) as f:
                    raw = json.load(f)
                expected = raw['test'][ti].get('output')
                if expected is None:
                    task_correct = False
                    continue
                
                # pass@2: any of the attempts correct?
                any_correct = any(grid_eq(pred, expected) for pred in test_preds)
                if not any_correct:
                    task_correct = False
            
            if task_attempted:
                attempted += 1
            if task_correct and task_attempted:
                correct += 1
            
            m = result.method
            methods[m] = methods.get(m, 0) + 1
            
            elapsed = time.time() - start
            speed = elapsed / (i + 1)
            
            status = '✓' if (task_correct and task_attempted) else '✗'
            print(f"  [{i+1}/{total}] {status} {speed:.1f}s/t {task.task_id} method={m}")
            
            if verbose and task_attempted:
                for atom in result.transforms_used[:1]:
                    print(f"    → {atom.explanation}")
            
            if (i + 1) % 50 == 0:
                pct = correct * 100 / (i + 1) if (i + 1) > 0 else 0
                print(f"[{i+1}/{total}] {correct}/{i+1} ({pct:.1f}%) attempted={attempted} ETA {speed*(total-i-1):.0f}s")
        
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{total}] ERROR {tf}: {e}")
    
    elapsed = time.time() - start
    pct = correct * 100 / total if total > 0 else 0
    att_pct = attempted * 100 / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"ARC-AGI-2 Evaluation ({split})")
    print(f"{'='*60}")
    print(f"Score: {correct}/{total} = {pct:.1f}%")
    print(f"Attempted: {attempted}/{total} = {att_pct:.1f}%")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s/task)")
    print(f"\nMethods:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {m}: {cnt}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'split': split,
        'score': correct,
        'total': total,
        'pct': pct,
        'attempted': attempted,
        'errors': errors,
        'elapsed_s': elapsed,
        'methods': methods,
    }
    out_path = f'arc_eval_{split}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/tmp/arc-agi-2/data')
    parser.add_argument('--split', default='training', choices=['training', 'evaluation'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.split, args.limit, args.verbose)
