"""
arc/eval_cross.py — Evaluate Verantyx Cross Solver on ARC-AGI-2

Usage:
  python3 -m arc.eval_cross [--data-dir /path] [--split training|evaluation] [--limit N]
"""

import os
import sys
import time
import argparse
from arc.cross_solver import solve_task_cross


def evaluate(data_dir: str, split: str = "training", limit: int = 0):
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
    candidates_total = 0
    verified_total = 0
    
    start = time.time()
    
    for i, tf in enumerate(task_files):
        path = os.path.join(task_dir, tf)
        try:
            result = solve_task_cross(path)
            
            if result['attempted']:
                attempted += 1
            if result['correct']:
                correct += 1
            
            m = result['method']
            methods[m] = methods.get(m, 0) + 1
            candidates_total += result['n_candidates']
            verified_total += result['n_verified']
            
            elapsed = time.time() - start
            speed = elapsed / (i + 1)
            
            status = '✓' if result['correct'] else '✗'
            task_id = tf.replace('.json', '')
            
            extra = f"cand={result['n_candidates']} ver={result['n_verified']}"
            if result['correct']:
                extra += f" rule={result['rule']}"
            
            print(f"  [{i+1}/{total}] {status} {speed:.2f}s/t {task_id} {extra}")
            
            if (i + 1) % 50 == 0:
                pct = correct * 100 / (i + 1)
                eta = speed * (total - i - 1)
                print(f"[{i+1}/{total}] {correct}/{i+1} ({pct:.1f}%) att={attempted} ETA={eta:.0f}s")
        
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR {tf}: {e}")
    
    elapsed = time.time() - start
    pct = correct * 100 / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"ARC-AGI-2 Cross Solver ({split})")
    print(f"{'='*60}")
    print(f"Score: {correct}/{total} = {pct:.1f}%")
    print(f"Attempted: {attempted}/{total}")
    print(f"Candidates generated: {candidates_total} ({candidates_total/total:.0f}/task avg)")
    print(f"Verified programs: {verified_total}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s/task)")
    print(f"\nMethods:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {m}: {cnt}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/tmp/arc-agi-2/data')
    parser.add_argument('--split', default='training')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.split, args.limit)
