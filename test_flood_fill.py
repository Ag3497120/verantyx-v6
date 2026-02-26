#!/usr/bin/env python3
"""Test flood_fill module on specific tasks"""

import json
import sys
from arc.grid import grid_eq
from arc.cross_engine import solve_cross_engine


def test_task(task_id: str, data_dir: str = "/tmp/arc-agi-2/data/training"):
    """Test a single task"""
    task_path = f"{data_dir}/{task_id}.json"

    try:
        with open(task_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Task {task_id} not found at {task_path}")
        return False

    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_inputs = [ex['input'] for ex in data['test']]
    test_outputs = [ex.get('output') for ex in data['test']]

    print(f"\n{'='*60}")
    print(f"Testing task: {task_id}")
    print(f"{'='*60}")

    predictions, verified = solve_cross_engine(train, test_inputs)

    if not verified:
        print("❌ No solution found")
        return False

    print(f"\n✅ Solution found:")
    for kind, prog in verified[:3]:
        if kind == 'cross':
            print(f"  - Cross piece: {prog.name}")
        elif hasattr(prog, 'name'):
            print(f"  - {kind}: {prog.name}")
        else:
            print(f"  - {kind}")

    # Check test results
    all_correct = True
    for i, (preds, expected) in enumerate(zip(predictions, test_outputs)):
        if expected is None:
            continue
        if not preds:
            print(f"  Test {i}: ❌ No prediction")
            all_correct = False
        elif any(grid_eq(p, expected) for p in preds):
            print(f"  Test {i}: ✅ Correct")
        else:
            print(f"  Test {i}: ❌ Wrong")
            all_correct = False

    return all_correct


if __name__ == "__main__":
    tasks = [
        "0e671a1a",
        "1e32b0e9",
        "0d87d2a6",
        "1478ab18",
        "1f0c79e5",
    ]

    results = {}
    for task_id in tasks:
        results[task_id] = test_task(task_id)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for task_id, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {task_id}: {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tasks passed")
