#!/usr/bin/env python3
"""Debug task 22208ba4"""

import json
from arc.conditional_transform import (
    learn_conditional_object_transform,
    apply_conditional_object_transform
)
from arc.grid import grid_eq

path = "/tmp/arc-agi-2/data/training/22208ba4.json"
with open(path) as f:
    data = json.load(f)

train_pairs = [(ex['input'], ex['output']) for ex in data['train']]

# Pretty print first pair
inp, out = train_pairs[0]
print("Input (16x16):")
for row in inp:
    print(''.join(str(c) for c in row))

print("\nOutput (16x16):")
for row in out:
    print(''.join(str(c) for c in row))

print("\nPattern analysis:")
print("Input has 2 colored cells at (0,0) and (0,15) -> corners")
print("Output has same cells moved to (1,1) and (1,14) -> moved diagonally inward")

# Try to learn
rule = learn_conditional_object_transform(train_pairs)
print(f"\nLearned rule: {rule}")

if rule:
    # Test on first pair
    result = apply_conditional_object_transform(inp, rule)
    if result and grid_eq(result, out):
        print("✅ First pair matches!")
    else:
        print("❌ First pair doesn't match")
        if result:
            print("\nResult:")
            for row in result[:5]:
                print(''.join(str(c) for c in row[:20]))
