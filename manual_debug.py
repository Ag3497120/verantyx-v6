#!/usr/bin/env python3
"""Manual debug to understand task pattern"""

import json
from arc.grid import grid_shape, grid_eq


def analyze_task(task_id: str):
    task_path = f"/tmp/arc-agi-2/data/training/{task_id}.json"

    with open(task_path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    for idx, ex in enumerate(data['train'][:1]):  # Just first example
        inp = ex['input']
        out = ex['output']

        h, w = grid_shape(inp)
        print(f"\nExample {idx}:")
        print(f"  Size: {h}x{w}")

        # Find colored dots in input
        dots = []
        for r in range(h):
            for c in range(w):
                if inp[r][c] != 0:
                    dots.append((r, c, inp[r][c]))

        print(f"  Input dots: {dots}")

        # Find what changed
        changes = []
        for r in range(h):
            for c in range(w):
                if inp[r][c] != out[r][c]:
                    changes.append((r, c, inp[r][c], out[r][c]))

        print(f"  Total changes: {len(changes)}")
        if len(changes) <= 30:
            print(f"  Changes: {changes[:10]}...")

        # Find unique colors
        inp_colors = set()
        out_colors = set()
        for row in inp:
            inp_colors.update(row)
        for row in out:
            out_colors.update(row)

        print(f"  Input colors: {sorted(inp_colors)}")
        print(f"  Output colors: {sorted(out_colors)}")
        print(f"  New colors in output: {sorted(out_colors - inp_colors)}")


if __name__ == "__main__":
    for task_id in ["0e671a1a", "1e32b0e9", "1478ab18"]:
        analyze_task(task_id)
