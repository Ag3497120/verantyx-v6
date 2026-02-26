#!/usr/bin/env python3
"""Verify flood_fill works on found task"""

import json
from arc.flood_fill import learn_flood_fill_region, apply_flood_fill_region
from arc.grid import grid_eq, grid_to_str


def verify_task(task_id: str):
    task_path = f"/tmp/arc-agi-2/data/training/{task_id}.json"

    with open(task_path) as f:
        data = json.load(f)

    train = [(ex['input'], ex['output']) for ex in data['train']]
    test = [(ex['input'], ex.get('output')) for ex in data['test']]

    print(f"Task: {task_id}")
    print(f"{'='*60}\n")

    # Learn rule
    rule = learn_flood_fill_region(train)
    print(f"Learned rule: {rule}\n")

    # Test on training
    print("Training pairs:")
    for i, (inp, expected) in enumerate(train):
        result = apply_flood_fill_region(inp, rule)
        match = grid_eq(result, expected) if result else False
        status = "✅" if match else "❌"
        print(f"  Pair {i}: {status}")

    # Test on test
    print("\nTest pairs:")
    for i, (inp, expected) in enumerate(test):
        result = apply_flood_fill_region(inp, rule)
        if expected:
            match = grid_eq(result, expected) if result else False
            status = "✅" if match else "❌"
            print(f"  Pair {i}: {status}")
        else:
            print(f"  Pair {i}: Generated prediction")


if __name__ == "__main__":
    verify_task("00d62c1b")
