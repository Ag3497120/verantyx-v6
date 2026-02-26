#!/usr/bin/env python3
"""Analyze unsolved ARC tasks to find pattern categories"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Parse the log to find solved tasks
solved = set()
log_path = "arc_v53_full.log"
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            if "✓" in line:
                # Format: [1/1000] ✓ 0.24s/t 00576224 ver=2 rule=...
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "✓" and i + 2 < len(parts):
                        # Task ID is 2 positions after ✓
                        task_id = parts[i + 2]
                        if len(task_id) == 8:  # ARC task IDs are 8 chars
                            solved.add(task_id)

print(f"Found {len(solved)} solved tasks")

# Load all training tasks
data_dir = Path("/tmp/arc-agi-2/data/training")
all_tasks = list(data_dir.glob("*.json"))
print(f"Total training tasks: {len(all_tasks)}")

unsolved = []
for task_file in all_tasks:
    task_id = task_file.stem
    if task_id not in solved:
        with open(task_file) as f:
            task_data = json.load(f)
        unsolved.append((task_id, task_data))

print(f"Unsolved tasks: {len(unsolved)}")

# Analyze patterns
def analyze_task(task_id, task_data):
    """Quick pattern analysis"""
    train_pairs = task_data.get("train", [])
    if not train_pairs:
        return None

    patterns = []

    # Check for size change patterns
    inp = train_pairs[0]["input"]
    out = train_pairs[0]["output"]
    ih, iw = len(inp), len(inp[0]) if inp else 0
    oh, ow = len(out), len(out[0]) if out else 0

    # Pattern 1: NxN scaling (grow)
    if oh > ih and ow > iw:
        if oh % ih == 0 and ow % iw == 0:
            scale_h = oh // ih
            scale_w = ow // iw
            if scale_h == scale_w and scale_h > 1:
                patterns.append(f"scale_{scale_h}x")

    # Pattern 2: Fixed output size (shrink/extract)
    all_same_output = all(
        len(p["output"]) == oh and len(p["output"][0]) == ow
        for p in train_pairs
    )
    all_diff_input = len(set(
        (len(p["input"]), len(p["input"][0]) if p["input"] else 0)
        for p in train_pairs
    )) > 1

    if all_same_output and all_diff_input:
        patterns.append(f"fixed_output_{oh}x{ow}")
    elif all_same_output and oh <= 5 and ow <= 5:
        patterns.append(f"small_fixed_output_{oh}x{ow}")

    # Pattern 3: Same size (object manipulation)
    if ih == oh and iw == ow:
        # Check if objects are present
        inp_colors = set()
        out_colors = set()
        for row in inp:
            inp_colors.update(row)
        for row in out:
            out_colors.update(row)

        if len(inp_colors) > 2 or len(out_colors) > 2:
            patterns.append("same_size_multi_color")

    return patterns

# Categorize
categories = defaultdict(list)
for task_id, task_data in unsolved:
    patterns = analyze_task(task_id, task_data)
    if patterns:
        for p in patterns:
            categories[p].append(task_id)

# Print categories sorted by frequency
print("\n=== UNSOLVED TASK CATEGORIES ===")
for category, tasks in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"{category}: {len(tasks)} tasks")
    if len(tasks) <= 10:
        print(f"  Tasks: {', '.join(tasks[:10])}")
    else:
        print(f"  Sample tasks: {', '.join(tasks[:5])}")

# Save detailed analysis
with open("unsolved_analysis.json", "w") as f:
    json.dump({
        "total_unsolved": len(unsolved),
        "categories": {k: v for k, v in categories.items()},
        "unsolved_ids": [tid for tid, _ in unsolved]
    }, f, indent=2)

print("\n✓ Saved detailed analysis to unsolved_analysis.json")
