"""eval_repair.py — Cross Engine + 修復 + 6軸Cross 統合"""
import json, sys, time, re
from pathlib import Path
from arc.eval_cross_engine import solve_task_engine
from arc.cross_engine import solve_cross_engine
from arc.cross_repair import repair_prediction
from arc.cross_vote import cross_vote_solve
from arc.grid import grid_eq


def eval_repair(data_dir, split="evaluation"):
    task_dir = Path(data_dir) / split
    tasks = sorted(task_dir.glob("*.json"))
    total = len(tasks)
    correct = 0; by_ce = 0; by_repair = 0; by_c6 = 0

    for i, tf in enumerate(tasks):
        tid = tf.stem
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        to = task['test'][0].get('output')
        
        # Phase 1: Cross Engine
        preds, verified = solve_cross_engine(tp, [ti])
        
        # Check direct solve
        if preds and preds[0]:
            for p in preds[0]:
                if to and grid_eq(p, to):
                    correct += 1; by_ce += 1
                    print(f"  [{i+1}/{total}] ✓ {tid} [CE]")
                    break
            else:
                # Phase 2: Repair each prediction
                solved = False
                for p in preds[0][:3]:
                    repaired = repair_prediction(ti, p, tp)
                    if to and grid_eq(repaired, to):
                        correct += 1; by_repair += 1
                        print(f"  [{i+1}/{total}] ✓ {tid} [REPAIR]")
                        solved = True
                        break
                
                if not solved:
                    # Phase 3: 6-Axis Cross
                    r = cross_vote_solve(tp, ti)
                    if r and to and grid_eq(r, to):
                        correct += 1; by_c6 += 1
                        print(f"  [{i+1}/{total}] ✓ {tid} [C6]")
                    else:
                        print(f"  [{i+1}/{total}] ✗ {tid}")
        else:
            # No CE prediction — try 6-axis
            r = cross_vote_solve(tp, ti)
            if r and to and grid_eq(r, to):
                correct += 1; by_c6 += 1
                print(f"  [{i+1}/{total}] ✓ {tid} [C6]")
            else:
                print(f"  [{i+1}/{total}] ✗ {tid}")

    print(f"\n{'='*60}")
    print(f"Score: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  CE: {by_ce}, Repair: {by_repair}, C6: {by_c6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    eval_repair("/tmp/arc-agi-2/data", split)
