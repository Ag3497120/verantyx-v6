#!/usr/bin/env python3
"""Test stamping pattern tasks"""

import json
from arc.grid import grid_eq, grid_shape, most_common_color
from arc.objects import detect_objects

def test_2c737e39():
    """Task 2c737e39 - object stamping"""
    path = "/tmp/arc-agi-2/data/training/2c737e39.json"
    with open(path) as f:
        data = json.load(f)

    train_pairs = [(ex['input'], ex['output']) for ex in data['train']]

    print("Task 2c737e39 - Object Stamping Pattern")
    print("="*60)

    for i, (inp, out) in enumerate(train_pairs):
        print(f"\nTrain {i}:")
        print(f"Input shape: {grid_shape(inp)}")
        print(f"Output shape: {grid_shape(out)}")

        bg = most_common_color(inp)
        print(f"Background: {bg}")

        # Find objects and marker positions
        objs = detect_objects(inp, bg)
        print(f"Objects detected: {len(objs)}")

        # Find isolated markers (color 5 in this case)
        for r, row in enumerate(inp):
            for c, val in enumerate(row):
                if val == 5:
                    print(f"  Marker at ({r}, {c})")

def test_50c07299():
    """Task 50c07299 - diagonal line pattern"""
    path = "/tmp/arc-agi-2/data/training/50c07299.json"
    with open(path) as f:
        data = json.load(f)

    train_pairs = [(ex['input'], ex['output']) for ex in data['train']]

    print("\n\nTask 50c07299 - Diagonal Pattern")
    print("="*60)

    for i, (inp, out) in enumerate(train_pairs):
        print(f"\nTrain {i}:")
        print(f"Input shape: {grid_shape(inp)}")
        print(f"Output shape: {grid_shape(out)}")

        # Print first few rows to see pattern
        print("Input (first 3 rows):")
        for row in inp[:3]:
            print('  ' + ''.join(str(c) for c in row))

        print("Output (first 3 rows):")
        for row in out[:3]:
            print('  ' + ''.join(str(c) for c in row))

if __name__ == '__main__':
    test_2c737e39()
    test_50c07299()
