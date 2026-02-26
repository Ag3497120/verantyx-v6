#!/usr/bin/env python3
"""Test conditional_transform on specific tasks"""

import json
from arc.cross_engine import solve_cross_engine
from arc.grid import grid_eq


def test_task(task_id: str):
    """Test a specific task"""
    path = f"/tmp/arc-agi-2/data/training/{task_id}.json"
    with open(path) as f:
        data = json.load(f)

    train_pairs = [(ex['input'], ex['output']) for ex in data['train']]
    test_pairs = [(ex['input'], ex.get('output')) for ex in data['test']]

    print(f"\n{'='*60}")
    print(f"Testing task: {task_id}")
    print(f"{'='*60}")

    # Try to solve
    test_inputs = [inp for inp, _ in test_pairs]
    predictions, verified = solve_cross_engine(train_pairs, test_inputs)

    if not verified:
        print("❌ No solution found")
        return False

    # Use first verified solution
    _, piece = verified[0]
    print(f"✅ Found solution: {piece.name}")

    # Test on training
    train_ok = True
    for i, (inp, out) in enumerate(train_pairs):
        pred = piece.apply(inp)
        if pred is None or not grid_eq(pred, out):
            print(f"  ❌ Train {i}: FAIL")
            train_ok = False
        else:
            print(f"  ✅ Train {i}: PASS")

    # Test on test
    test_ok = True
    for i, (inp, out) in enumerate(test_pairs):
        pred = piece.apply(inp)
        if out is not None:
            if pred is None or not grid_eq(pred, out):
                print(f"  ❌ Test {i}: FAIL")
                test_ok = False
            else:
                print(f"  ✅ Test {i}: PASS")
        else:
            if pred is not None:
                print(f"  📋 Test {i}: Generated prediction")
            else:
                print(f"  ❌ Test {i}: No prediction")

    return train_ok and (test_ok or all(out is None for _, out in test_pairs))


if __name__ == '__main__':
    tasks = ['22208ba4', '2c737e39', '50c07299']

    results = {}
    for task_id in tasks:
        try:
            results[task_id] = test_task(task_id)
        except Exception as e:
            print(f"❌ Error on {task_id}: {e}")
            results[task_id] = False

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for task_id, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{task_id}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} ({100*passed//total if total > 0 else 0}%)")
