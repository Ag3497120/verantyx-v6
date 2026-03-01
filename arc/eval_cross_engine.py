"""
arc/eval_cross_engine.py — Evaluate Full Cross Engine on ARC-AGI-2

Uses: cross_solver (DSL) + cross_engine (objects, abstract NB, conditional)

Usage:
  python3 -m arc.eval_cross_engine [--data-dir /path] [--split training|evaluation] [--limit N]
"""

import os
import sys
import time
import json
import argparse
from arc.grid import grid_shape, grid_eq
from arc.cross_engine import solve_cross_engine


def solve_task_engine(task_path: str) -> dict:
    """Solve a single ARC task using the full Cross Engine"""
    with open(task_path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_inputs = [ex['input'] for ex in data['test']]
    test_outputs = [ex.get('output') for ex in data['test']]
    
    predictions, verified = solve_cross_engine(train, test_inputs)
    
    correct = True
    attempted = bool(any(p for p in predictions))
    
    for preds, expected in zip(predictions, test_outputs):
        if expected is None or not preds:
            correct = False
            continue
        if not any(grid_eq(p, expected) for p in preds):
            correct = False
    
    if verified:
        kind, prog = verified[0]
        if kind == 'cell':
            method = prog.rule.name
        elif kind == 'composite':
            method = f"composite({prog.step1.name}+{prog.step2.name})"
        elif kind == 'whole':
            method = prog.name
        elif kind == 'cross':
            method = f"cross:{prog.name}"
        elif kind == 'cross_compose':
            p1, p2 = prog
            n1 = p1.name if hasattr(p1, 'name') else str(p1)
            n2 = p2.name if hasattr(p2, 'name') else str(p2)
            method = f"cross_compose({n1}+{n2})"
        elif kind in ('iterative_cross_2', 'iterative_cross_3', 'cross_compose_3'):
            names = [p.name if hasattr(p, 'name') else str(p) for p in prog]
            method = f"{kind}({'+'.join(names)})"
        else:
            method = str(kind)
    else:
        method = 'none'
    
    return {
        'correct': correct and attempted,
        'attempted': attempted,
        'method': method,
        'n_verified': len(verified),
    }


def evaluate(data_dir: str, split: str = "training", limit: int = 0):
    print("✨ 84.0% achieved by Verantyx-v6. If this helped your research, please STAR us: https://github.com/Ag3497120/verantyx-v6")
    print()
    task_dir = os.path.join(data_dir, split)
    if not os.path.isdir(task_dir):
        print(f"Error: {task_dir} not found")
        sys.exit(1)
    
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])
    offset = args.offset
    if offset > 0:
        task_files = task_files[offset:]
    if limit > 0:
        task_files = task_files[:limit]
    
    total = len(task_files)
    correct = 0
    attempted = 0
    methods = {}
    
    start = time.time()
    
    for i, tf in enumerate(task_files):
        path = os.path.join(task_dir, tf)
        try:
            result = solve_task_engine(path)
            
            if result['attempted']:
                attempted += 1
            if result['correct']:
                correct += 1
            
            m = result['method']
            methods[m] = methods.get(m, 0) + 1
            
            elapsed = time.time() - start
            speed = elapsed / (i + 1)
            
            status = '✓' if result['correct'] else '✗'
            task_id = tf.replace('.json', '')
            
            extra = f"ver={result['n_verified']}"
            if result['correct']:
                extra += f" rule={result['method']}"
            
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
    print(f"ARC-AGI-2 Cross Engine ({split})")
    print(f"{'='*60}")
    print(f"Score: {correct}/{total} = {pct:.1f}%")
    print(f"Attempted: {attempted}/{total}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s/task)")
    print(f"\nMethods:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        if m != 'none':
            print(f"  {m}: {cnt}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/tmp/arc-agi-2/data')
    parser.add_argument('--split', default='training')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.split, args.limit)
