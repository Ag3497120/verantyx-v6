#!/usr/bin/env python3
"""Deep analysis of scale/grow tasks"""

import json
import numpy as np
from pathlib import Path

def analyze_scale_task(task_id):
    """Analyze a specific scaling task"""
    task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
    with open(task_path) as f:
        data = json.load(f)

    train = data["train"]
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    for idx, pair in enumerate(train):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])

        ih, iw = inp.shape
        oh, ow = out.shape

        print(f"\nTrain pair {idx}:")
        print(f"  Input: {ih}x{iw}, Output: {oh}x{ow}")
        print(f"  Scale factor: {oh//ih if ih > 0 else '?'}x{ow//iw if iw > 0 else '?'}")

        if oh % ih == 0 and ow % iw == 0:
            scale = oh // ih
            print(f"  Block size: {scale}x{scale}")

            # Check what mapping exists
            # For each input cell, what does the output block look like?
            color_patterns = {}

            for r in range(ih):
                for c in range(iw):
                    inp_color = int(inp[r, c])
                    block = out[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                    block_tuple = tuple(block.flatten().tolist())

                    if inp_color not in color_patterns:
                        color_patterns[inp_color] = []
                    color_patterns[inp_color].append(block_tuple)

            # Check if patterns are consistent
            print(f"  Color → Block pattern analysis:")
            for color, patterns in sorted(color_patterns.items()):
                unique_patterns = set(patterns)
                if len(unique_patterns) == 1:
                    pattern = np.array(list(unique_patterns)[0]).reshape(scale, scale)
                    print(f"    Color {color} → consistent pattern: {pattern.flatten().tolist()}")
                else:
                    print(f"    Color {color} → INCONSISTENT ({len(unique_patterns)} different patterns)")
                    # Show first few
                    for i, p in enumerate(list(unique_patterns)[:3]):
                        print(f"      Pattern {i}: {list(p)}")

        # Show the grids
        print(f"\n  Input grid:")
        print(inp)
        print(f"\n  Output grid:")
        print(out)

# Analyze scale_3x tasks
scale_3x_tasks = ['c3e719e8', '310f3251', 'c92b942c', '8e2edd66', '15696249']
print("ANALYZING SCALE 3X TASKS")
print("="*60)

for task_id in scale_3x_tasks[:3]:  # First 3
    try:
        analyze_scale_task(task_id)
    except Exception as e:
        print(f"Error analyzing {task_id}: {e}")

# Analyze scale_2x tasks
scale_2x_tasks = ['f0afb749', '539a4f51', 'fb791726', '10fcaaa3']
print("\n\n")
print("ANALYZING SCALE 2X TASKS")
print("="*60)

for task_id in scale_2x_tasks[:2]:  # First 2
    try:
        analyze_scale_task(task_id)
    except Exception as e:
        print(f"Error analyzing {task_id}: {e}")
