"""
ARC モジュールのテスト
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from arc.grid_ir import GridDecomposer, decompose_pair
from arc.arc_cegis import ARCCEGISLoop, solve_arc_task

# ── テスト1: GridDecomposer ──
print("=" * 50)
print("Test 1: GridDecomposer")
print("=" * 50)

grid = [
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
]
dec = GridDecomposer()
ir = dec.decompose(grid)
print(f"  Size: {ir.height}x{ir.width}")
print(f"  Colors: {ir.colors}, BG: {ir.background_color}")
print(f"  Objects: {len(ir.objects)}")
for obj in ir.objects:
    print(f"    obj_{obj.obj_id}: color={obj.color}, size={obj.size}, bbox={obj.bbox}, rect={obj.is_rectangular()}, line={obj.is_line()}")
print(f"  Symmetries: {[s.kind for s in ir.symmetries]}")
print(f"  Patterns: {[p.kind for p in ir.patterns]}")
print()

# ── テスト2: 回転問題 ──
print("=" * 50)
print("Test 2: Rotation task (rotate_90)")
print("=" * 50)

task_rotate = {
    "train": [
        {
            "input":  [[1, 2], [3, 4]],
            "output": [[3, 1], [4, 2]],
        },
        {
            "input":  [[5, 6], [7, 8]],
            "output": [[7, 5], [8, 6]],
        },
    ],
    "test": [{"input": [[9, 0], [1, 2]]}],
}

result = solve_arc_task(task_rotate)
print(f"  Prediction: {result}")
print(f"  Expected:   [[1, 9], [2, 0]]")
print(f"  Correct:    {result == [[1, 9], [2, 0]]}")
print()

# ── テスト3: 色スワップ ──
print("=" * 50)
print("Test 3: Color swap (1 <-> 2)")
print("=" * 50)

task_swap = {
    "train": [
        {
            "input":  [[1, 0, 2], [0, 1, 0]],
            "output": [[2, 0, 1], [0, 2, 0]],
        },
        {
            "input":  [[2, 2, 1], [1, 0, 2]],
            "output": [[1, 1, 2], [2, 0, 1]],
        },
    ],
    "test": [{"input": [[1, 1, 0], [2, 0, 1]]}],
}

result = solve_arc_task(task_swap)
print(f"  Prediction: {result}")
print(f"  Expected:   [[2, 2, 0], [1, 0, 2]]")
print(f"  Correct:    {result == [[2, 2, 0], [1, 0, 2]]}")
print()

# ── テスト4: 上下反転 ──
print("=" * 50)
print("Test 4: Flip vertical")
print("=" * 50)

task_flip = {
    "train": [
        {
            "input":  [[1, 0], [0, 2], [3, 0]],
            "output": [[3, 0], [0, 2], [1, 0]],
        },
        {
            "input":  [[4, 5], [6, 7], [8, 9]],
            "output": [[8, 9], [6, 7], [4, 5]],
        },
    ],
    "test": [{"input": [[1, 2], [3, 4], [5, 6]]}],
}

result = solve_arc_task(task_flip)
print(f"  Prediction: {result}")
print(f"  Expected:   [[5, 6], [3, 4], [1, 2]]")
print(f"  Correct:    {result == [[5, 6], [3, 4], [1, 2]]}")
print()

# ── テスト5: スケール2x ──
print("=" * 50)
print("Test 5: Scale 2x")
print("=" * 50)

task_scale = {
    "train": [
        {
            "input":  [[1, 2], [3, 0]],
            "output": [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]],
        },
    ],
    "test": [{"input": [[5, 6], [7, 8]]}],
}

result = solve_arc_task(task_scale)
expected = [[5, 5, 6, 6], [5, 5, 6, 6], [7, 7, 8, 8], [7, 7, 8, 8]]
print(f"  Prediction: {result}")
print(f"  Expected:   {expected}")
print(f"  Correct:    {result == expected}")
print()

# ── テスト6: 実ARC問題 (縦縞パターン繰り返し) ──
print("=" * 50)
print("Test 6: Real ARC pattern (column repeat)")
print("=" * 50)

# 0a938d79.json の簡略版
task_real = {
    "train": [
        {
            "input":  [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0]],
            "output": [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2],
                        [0, 0, 0, 0, 0, 2, 0, 8, 0, 2],
                        [0, 0, 0, 0, 0, 2, 0, 8, 0, 2],
                        [0, 0, 0, 0, 0, 2, 0, 8, 0, 2]],
        },
    ],
    "test": [{"input": [[0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 3, 0, 0, 0]]}],
}

result = solve_arc_task(task_real)
if result:
    for row in result:
        print(f"  {row}")
else:
    print("  解けなかった")
print()

print("=" * 50)
print("テスト完了")
