#!/usr/bin/env python3
"""Find line/ray extension tasks in unsolved set"""

import json
import numpy as np
from pathlib import Path

# Load unsolved list
with open("unsolved_analysis.json") as f:
    data = json.load(f)

unsolved_ids = data["unsolved_ids"]

def has_line_ray_pattern(task_id):
    """Check if task has line/ray extension pattern"""
    task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
    with open(task_path) as f:
        task_data = json.load(f)

    train = task_data["train"]

    for pair in train:
        inp = np.array(pair["input"])
        out = np.array(pair["output"])

        # Must be same size
        if inp.shape != out.shape:
            return False

        # Check if output has more non-zero cells than input
        inp_nonzero = np.count_nonzero(inp)
        out_nonzero = np.count_nonzero(out)

        if out_nonzero <= inp_nonzero:
            return False

        # Check for line patterns (horizontal/vertical runs)
        h, w = out.shape
        has_lines = False

        for r in range(h):
            # Check horizontal runs
            for c in range(w):
                if out[r, c] != 0 and inp[r, c] == 0:
                    # This cell was added
                    # Check if it's part of a horizontal or vertical line
                    left_colors = set()
                    right_colors = set()
                    up_colors = set()
                    down_colors = set()

                    # Check left
                    for c2 in range(c):
                        if out[r, c2] != 0:
                            left_colors.add(out[r, c2])

                    # Check right
                    for c2 in range(c + 1, w):
                        if out[r, c2] != 0:
                            right_colors.add(out[r, c2])

                    # If there are colored cells on both sides, might be a line
                    if left_colors and right_colors:
                        has_lines = True
                        break

                    # Check vertical
                    for r2 in range(r):
                        if out[r2, c] != 0:
                            up_colors.add(out[r2, c])

                    for r2 in range(r + 1, h):
                        if out[r2, c] != 0:
                            down_colors.add(out[r2, c])

                    if up_colors and down_colors:
                        has_lines = True
                        break

            if has_lines:
                break

        if not has_lines:
            return False

    return True

def has_interior_fill_pattern(task_id):
    """Check if task fills object interiors"""
    task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
    with open(task_path) as f:
        task_data = json.load(f)

    train = task_data["train"]

    for pair in train:
        inp = np.array(pair["input"])
        out = np.array(pair["output"])

        # Must be same size
        if inp.shape != out.shape:
            return False

        # Check if output has more non-zero cells
        inp_nonzero = np.count_nonzero(inp)
        out_nonzero = np.count_nonzero(out)

        if out_nonzero <= inp_nonzero:
            return False

        # Check if added cells are "inside" existing shapes
        # Look for enclosed regions
        diff = (out != 0).astype(int) - (inp != 0).astype(int)
        added = diff == 1

        if not added.any():
            return False

    return True

# Scan same_size tasks
same_size_tasks = [tid for tid in unsolved_ids
                   if tid in data["categories"].get("same_size_multi_color", [])]

print(f"Scanning {len(same_size_tasks)} same-size tasks for patterns...")

line_ray_tasks = []
interior_fill_tasks = []

for i, task_id in enumerate(same_size_tasks[:200]):  # Check first 200
    if i % 50 == 0:
        print(f"Progress: {i}/{len(same_size_tasks[:200])}")

    try:
        if has_line_ray_pattern(task_id):
            line_ray_tasks.append(task_id)
        elif has_interior_fill_pattern(task_id):
            interior_fill_tasks.append(task_id)
    except Exception as e:
        pass

print(f"\nLine/Ray extension candidates: {len(line_ray_tasks)}")
print(f"Sample: {line_ray_tasks[:10]}")

print(f"\nInterior fill candidates: {len(interior_fill_tasks)}")
print(f"Sample: {interior_fill_tasks[:10]}")

# Save results
with open("pattern_candidates.json", "w") as f:
    json.dump({
        "line_ray": line_ray_tasks,
        "interior_fill": interior_fill_tasks
    }, f, indent=2)
