#!/usr/bin/env python3
"""Scan for tasks that match flood_fill patterns"""

import os
import json
from arc.flood_fill import learn_flood_fill_region, apply_flood_fill_region
from arc.grid import grid_eq


def scan_tasks(data_dir: str = "/tmp/arc-agi-2/data/training", limit: int = 50):
    """Scan tasks to find ones that match flood_fill patterns"""

    task_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:limit]

    matches = []

    for task_file in task_files:
        task_path = os.path.join(data_dir, task_file)
        task_id = task_file[:-5]

        try:
            with open(task_path) as f:
                data = json.load(f)

            train = [(ex['input'], ex['output']) for ex in data['train']]

            # Try to learn rule
            rule = learn_flood_fill_region(train)

            if rule is not None:
                # Verify on training pairs
                all_correct = True
                for inp, expected in train:
                    result = apply_flood_fill_region(inp, rule)
                    if result is None or not grid_eq(result, expected):
                        all_correct = False
                        break

                if all_correct:
                    matches.append((task_id, rule['type']))
                    print(f"✅ {task_id}: {rule['type']}")

        except Exception as e:
            pass

    print(f"\n{'='*60}")
    print(f"Found {len(matches)} matching tasks out of {len(task_files)}")
    for task_id, rule_type in matches:
        print(f"  - {task_id}: {rule_type}")


if __name__ == "__main__":
    scan_tasks(limit=100)
