#!/usr/bin/env python3
"""Debug grow primitive"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arc.grow_primitives import learn_grow_via_self_stamp
from arc.grid import most_common_color

task_id = 'c3e719e8'
task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
with open(task_path) as f:
    data = json.load(f)

train = [(p["input"], p["output"]) for p in data["train"]]

print(f"Testing task {task_id}")
print(f"Number of training pairs: {len(train)}")

# Manually check the pattern
for idx, (inp, out) in enumerate(train):
    ai = np.array(inp)
    ao = np.array(out)
    h, w = ai.shape
    oh, ow = ao.shape

    print(f"\nPair {idx}:")
    print(f"  Input shape: {ai.shape}")
    print(f"  Output shape: {ao.shape}")
    print(f"  Expected output shape for self-stamp: ({h*h}x{w*w})")

    if ao.shape != (h * h, w * w):
        print(f"  ❌ Output shape doesn't match self-stamp pattern!")
        continue

    print(f"  ✓ Output shape matches!")

    # Check most common color
    from collections import Counter
    c = Counter(ai.flatten().tolist())
    most_common = c.most_common(1)[0][0]
    print(f"  Most common color in input: {most_common}")

    # Check if pattern matches
    bg = most_common_color(out)
    print(f"  Background color in output: {bg}")

    expected = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if ai[r, c] == most_common:
                expected[r * h:(r + 1) * h, c * w:(c + 1) * w] = ai

    if np.array_equal(expected, ao):
        print(f"  ✓ Pattern matches 'most_common' strategy!")
    else:
        print(f"  ❌ Pattern doesn't match")
        # Show first mismatch
        diff = expected != ao
        if diff.any():
            r, c = np.where(diff)
            print(f"    First mismatch at ({r[0]}, {c[0]}): expected={expected[r[0], c[0]]}, actual={ao[r[0], c[0]]}")

print("\n\nNow testing learn function:")
result = learn_grow_via_self_stamp(train)
print(f"Learn result: {result}")
