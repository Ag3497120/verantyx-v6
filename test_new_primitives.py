#!/usr/bin/env python3
"""Test new primitives on sample tasks"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from arc.grow_primitives import (
    learn_grow_via_self_stamp, apply_grow_via_self_stamp,
    learn_grow_fixed_color_template, apply_grow_fixed_color_template,
)
from arc.line_ray_primitives import (
    learn_line_ray_from_objects, apply_line_ray_from_objects,
    learn_fill_object_interior, apply_fill_object_interior,
)
from arc.grid import grid_eq

def test_task(task_id, learn_fn, apply_fn, name):
    """Test a primitive on a task"""
    task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
    with open(task_path) as f:
        data = json.load(f)

    train = [(p["input"], p["output"]) for p in data["train"]]
    test = [(p["input"], p.get("output")) for p in data.get("test", [])]

    print(f"\n{'='*60}")
    print(f"Testing {name} on {task_id}")
    print(f"{'='*60}")

    # Learn
    try:
        params = learn_fn(train)
        if params is None:
            print(f"❌ Failed to learn pattern")
            return False

        print(f"✓ Learned pattern: {params.get('type', name)}")

        # Verify on training
        train_ok = 0
        for idx, (inp, out) in enumerate(train):
            result = apply_fn(inp, params)
            if result is not None and grid_eq(result, out):
                train_ok += 1
                print(f"  Train {idx}: ✓")
            else:
                print(f"  Train {idx}: ❌")

        if train_ok == len(train):
            print(f"✓ ALL {len(train)} training pairs match!")
            return True
        else:
            print(f"❌ Only {train_ok}/{len(train)} training pairs match")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test grow primitives
print("TESTING GROW PRIMITIVES")
print("="*60)

# Test self-stamp on scale tasks
scale_3x_tasks = ['c3e719e8']
for task_id in scale_3x_tasks:
    test_task(task_id, learn_grow_via_self_stamp, apply_grow_via_self_stamp, "grow_self_stamp")

# Test color template
color_template_tasks = ['f0afb749', '539a4f51']
for task_id in color_template_tasks[:1]:
    test_task(task_id, learn_grow_fixed_color_template, apply_grow_fixed_color_template, "grow_color_template")

# Test line ray primitives
print("\n\nTESTING LINE/RAY PRIMITIVES")
print("="*60)

line_ray_candidates = ['94414823', 'dc2e9a9d', 'baf41dbf']
for task_id in line_ray_candidates[:3]:
    test_task(task_id, learn_line_ray_from_objects, apply_line_ray_from_objects, "line_ray")

# Test fill interior
print("\n\nTESTING FILL INTERIOR PRIMITIVES")
print("="*60)

fill_candidates = ['5c0a986e', 'ec883f72']
for task_id in fill_candidates[:2]:
    test_task(task_id, learn_fill_object_interior, apply_fill_object_interior, "fill_interior")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
