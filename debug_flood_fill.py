#!/usr/bin/env python3
"""Debug flood_fill module"""

import json
from arc.flood_fill import learn_flood_fill_region, apply_flood_fill_region
from arc.grid import grid_eq


def debug_task(task_id: str, data_dir: str = "/tmp/arc-agi-2/data/training"):
    """Debug a single task"""
    task_path = f"{data_dir}/{task_id}.json"

    with open(task_path) as f:
        data = json.load(f)

    train = [(ex['input'], ex['output']) for ex in data['train']]

    print(f"\n{'='*60}")
    print(f"Debugging task: {task_id}")
    print(f"{'='*60}")
    print(f"Training pairs: {len(train)}")

    # Try to learn rule
    rule = learn_flood_fill_region(train)

    if rule is None:
        print("❌ Failed to learn rule")
        return False

    print(f"✅ Learned rule: {rule}")

    # Verify on training pairs
    all_correct = True
    for i, (inp, expected) in enumerate(train):
        result = apply_flood_fill_region(inp, rule)
        if result is None:
            print(f"  Train {i}: ❌ No result")
            all_correct = False
        elif grid_eq(result, expected):
            print(f"  Train {i}: ✅ Correct")
        else:
            print(f"  Train {i}: ❌ Wrong")
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

    for task_id in tasks:
        debug_task(task_id)
