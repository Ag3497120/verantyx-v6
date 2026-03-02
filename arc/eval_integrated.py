"""
arc/eval_integrated.py — Cross Engine + 6軸Cross 統合評価

1. Cross Engineで解く（既存のフル機能）
2. 解けなかった問題は6軸Crossソルバー（投票制）にフォールバック
"""

import json
import sys
import time
from pathlib import Path
from arc.eval_cross_engine import solve_task_engine
from arc.cross_vote import cross_vote_solve
from arc.grid import grid_eq


def evaluate_integrated(data_dir: str, split: str = "evaluation"):
    task_dir = Path(data_dir) / split
    if not task_dir.exists():
        task_dir = Path(data_dir)

    tasks = sorted(task_dir.glob("*.json"))
    total = len(tasks)
    correct = 0
    correct_ce = 0
    correct_c6 = 0
    attempted = 0

    t0 = time.time()
    for i, tf in enumerate(tasks):
        tid = tf.stem
        t1 = time.time()

        # Phase 1: Cross Engine
        result = solve_task_engine(str(tf))
        ce_ok = result.get('correct', False)

        if ce_ok:
            correct += 1
            correct_ce += 1
            elapsed = time.time() - t1
            print(f"  [{i+1}/{total}] ✓ {elapsed:.2f}s {tid} [CE] ver={result.get('ver', '?')} rule={result.get('method', '?')}")
            continue

        # Phase 2: 6-Axis Cross (vote)
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        
        solved_c6 = True
        for test_ex in task['test']:
            ti = test_ex['input']
            to = test_ex.get('output')
            
            r = cross_vote_solve(tp, ti)
            if r is None or to is None or not grid_eq(r, to):
                solved_c6 = False
                break

        if solved_c6:
            correct += 1
            correct_c6 += 1
            elapsed = time.time() - t1
            print(f"  [{i+1}/{total}] ✓ {elapsed:.2f}s {tid} [C6]")
        else:
            elapsed = time.time() - t1
            ver = result.get('ver', 0)
            print(f"  [{i+1}/{total}] ✗ {elapsed:.2f}s {tid} ver={ver}")

    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Integrated Score: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Cross Engine: {correct_ce}")
    print(f"  6-Axis Cross: {correct_c6}")
    print(f"  Time: {elapsed_total:.1f}s ({elapsed_total/total:.2f}s/task)")
    print(f"{'='*60}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    evaluate_integrated("/tmp/arc-agi-2/data", split)
