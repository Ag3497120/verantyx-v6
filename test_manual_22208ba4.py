#!/usr/bin/env python3
"""Manual test of 22208ba4 pattern"""

import json
from arc.grid import grid_eq, grid_shape

path = "/tmp/arc-agi-2/data/training/22208ba4.json"
with open(path) as f:
    data = json.load(f)

train_pairs = [(ex['input'], ex['output']) for ex in data['train']]

def move_edge_cells_inward(inp, bg=7):
    """Move cells on edges 1 step inward diagonally"""
    h, w = grid_shape(inp)
    result = [[bg] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                color = inp[r][c]
                new_r, new_c = r, c

                # On top edge -> move down
                if r == 0:
                    new_r = 1
                # On bottom edge -> move up
                elif r == h - 1:
                    new_r = h - 2

                # On left edge -> move right
                if c == 0:
                    new_c = 1
                # On right edge -> move left
                elif c == w - 1:
                    new_c = w - 2

                result[new_r][new_c] = color

    return result

# Test on all train pairs
all_match = True
for i, (inp, out) in enumerate(train_pairs):
    result = move_edge_cells_inward(inp)
    if grid_eq(result, out):
        print(f"✅ Train {i}: PASS")
    else:
        print(f"❌ Train {i}: FAIL")
        all_match = False

print(f"\nOverall: {'✅ ALL PASS' if all_match else '❌ FAILED'}")
