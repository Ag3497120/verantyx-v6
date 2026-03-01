#!/usr/bin/env python3
"""ARC-AGI2 batch solver - batch 2"""
import json
import os
import sys
import copy
from collections import Counter, defaultdict

DATA_DIR = "/private/tmp/arc-agi-2/data/training"
OUT_DIR = os.path.expanduser("~/verantyx_v6/synth_results")
BATCH_FILE = os.path.expanduser("~/verantyx_v6/agent_batch_2.json")

def load_task(task_id):
    path = os.path.join(DATA_DIR, f"{task_id}.json")
    with open(path) as f:
        return json.load(f)

def grids_equal(a, b):
    if len(a) != len(b): return False
    for ra, rb in zip(a, b):
        if ra != rb: return False
    return True

def test_func(func, examples):
    for ex in examples:
        try:
            result = func(copy.deepcopy(ex['input']))
            if not grids_equal(result, ex['output']):
                return False
        except:
            return False
    return True

def save_solution(task_id, code):
    path = os.path.join(OUT_DIR, f"{task_id}.py")
    with open(path, 'w') as f:
        f.write(code)
    print(f"  SAVED: {task_id}")

# ---- Common transforms ----

def identity(grid):
    return [row[:] for row in grid]

def rot90(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[R-1-c][r] for c in range(R)] for r in range(C)]

def rot180(grid):
    return [row[::-1] for row in grid[::-1]]

def rot270(grid):
    return rot90(rot90(rot90(grid)))

def flip_h(grid):
    return [row[::-1] for row in grid]

def flip_v(grid):
    return grid[::-1]

def flip_diag(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(R)] for c in range(C)]

def flip_adiag(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[R-1-c][C-1-r] for r in range(R)] for c in range(C)]

SIMPLE_TRANSFORMS = [
    ('identity', identity),
    ('rot90', rot90),
    ('rot180', rot180),
    ('rot270', rot270),
    ('flip_h', flip_h),
    ('flip_v', flip_v),
    ('flip_diag', flip_diag),
    ('flip_adiag', flip_adiag),
]

def try_simple(task_id, examples):
    for name, fn in SIMPLE_TRANSFORMS:
        if test_func(fn, examples):
            code = f"""import copy

def transform(grid):
    # {name}
"""
            if name == 'identity':
                code += "    return [row[:] for row in grid]\n"
            elif name == 'rot90':
                code += "    R, C = len(grid), len(grid[0])\n    return [[grid[R-1-c][r] for c in range(R)] for r in range(C)]\n"
            elif name == 'rot180':
                code += "    return [row[::-1] for row in grid[::-1]]\n"
            elif name == 'rot270':
                code += "    def rot90(g):\n        R, C = len(g), len(g[0])\n        return [[g[R-1-c][r] for c in range(R)] for r in range(C)]\n    return rot90(rot90(rot90(grid)))\n"
            elif name == 'flip_h':
                code += "    return [row[::-1] for row in grid]\n"
            elif name == 'flip_v':
                code += "    return grid[::-1]\n"
            elif name == 'flip_diag':
                code += "    R, C = len(grid), len(grid[0])\n    return [[grid[r][c] for r in range(R)] for c in range(C)]\n"
            elif name == 'flip_adiag':
                code += "    R, C = len(grid), len(grid[0])\n    return [[grid[R-1-c][C-1-r] for r in range(R)] for c in range(C)]\n"
            save_solution(task_id, code)
            return True
    return False

def get_color_map(inp, out):
    """Try to find a consistent color mapping."""
    mapping = {}
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            ci = inp[r][c]
            co = out[r][c]
            if ci in mapping:
                if mapping[ci] != co:
                    return None
            else:
                mapping[ci] = co
    return mapping

def try_color_map(task_id, examples):
    if not all(len(e['input']) == len(e['output']) and 
               len(e['input'][0]) == len(e['output'][0]) for e in examples):
        return False
    maps = []
    for ex in examples:
        m = get_color_map(ex['input'], ex['output'])
        if m is None:
            return False
        maps.append(m)
    # Merge maps
    merged = {}
    for m in maps:
        for k, v in m.items():
            if k in merged and merged[k] != v:
                return False
            merged[k] = v
    # Test
    def apply_map(grid):
        return [[merged.get(v, v) for v in row] for row in grid]
    if test_func(apply_map, examples):
        code = f"""def transform(grid):
    mapping = {merged}
    return [[mapping.get(v, v) for v in row] for row in grid]
"""
        save_solution(task_id, code)
        return True
    return False

def analyze_size_change(examples):
    changes = []
    for ex in examples:
        ih, iw = len(ex['input']), len(ex['input'][0])
        oh, ow = len(ex['output']), len(ex['output'][0])
        changes.append((ih, iw, oh, ow))
    return changes

def try_tiling(task_id, examples):
    """Check if output is input tiled N times."""
    for ry in [1,2,3,4]:
        for rx in [1,2,3,4]:
            if ry == 1 and rx == 1:
                continue
            def make_tiled(ry=ry, rx=rx):
                def f(grid):
                    rows = []
                    for _ in range(ry):
                        for r in grid:
                            rows.append(r * rx)
                    return rows
                return f
            fn = make_tiled(ry, rx)
            if test_func(fn, examples):
                code = f"""def transform(grid):
    rows = []
    for _ in range({ry}):
        for r in grid:
            rows.append(r * {rx})
    return rows
"""
                save_solution(task_id, code)
                return True
    return False

def find_flood_fill_components(grid, bg=0):
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    components = []
    def bfs(sr, sc, color):
        cells = []
        q = [(sr, sc)]
        visited[sr][sc] = True
        while q:
            r, c = q.pop()
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        return cells
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                cells = bfs(r, c, grid[r][c])
                components.append((grid[r][c], cells))
    return components

def try_gravity(task_id, examples):
    """Objects fall down."""
    def gravity_down(grid):
        R, C = len(grid), len(grid[0])
        result = [[0]*C for _ in range(R)]
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            non_bg = [v for v in col if v != 0]
            bg_count = col.count(0)
            new_col = [0]*bg_count + non_bg
            for r in range(R):
                result[r][c] = new_col[r]
        return result
    if test_func(gravity_down, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [[0]*C for _ in range(R)]
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        non_bg = [v for v in col if v != 0]
        bg_count = col.count(0)
        new_col = [0]*bg_count + non_bg
        for r in range(R):
            result[r][c] = new_col[r]
    return result
"""
        save_solution(task_id, code)
        return True

    def gravity_up(grid):
        R, C = len(grid), len(grid[0])
        result = [[0]*C for _ in range(R)]
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            non_bg = [v for v in col if v != 0]
            bg_count = col.count(0)
            new_col = non_bg + [0]*bg_count
            for r in range(R):
                result[r][c] = new_col[r]
        return result
    if test_func(gravity_up, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [[0]*C for _ in range(R)]
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        non_bg = [v for v in col if v != 0]
        bg_count = col.count(0)
        new_col = non_bg + [0]*bg_count
        for r in range(R):
            result[r][c] = new_col[r]
    return result
"""
        save_solution(task_id, code)
        return True

    def gravity_right(grid):
        result = []
        for row in grid:
            non_bg = [v for v in row if v != 0]
            bg = [0] * row.count(0)
            result.append(bg + non_bg)
        return result
    if test_func(gravity_right, examples):
        code = """def transform(grid):
    result = []
    for row in grid:
        non_bg = [v for v in row if v != 0]
        bg = [0] * row.count(0)
        result.append(bg + non_bg)
    return result
"""
        save_solution(task_id, code)
        return True

    def gravity_left(grid):
        result = []
        for row in grid:
            non_bg = [v for v in row if v != 0]
            bg = [0] * row.count(0)
            result.append(non_bg + bg)
        return result
    if test_func(gravity_left, examples):
        code = """def transform(grid):
    result = []
    for row in grid:
        non_bg = [v for v in row if v != 0]
        bg = [0] * row.count(0)
        result.append(non_bg + bg)
    return result
"""
        save_solution(task_id, code)
        return True
    return False

def try_outline(task_id, examples):
    """Fill interior of shapes, or outline objects."""
    def fill_interior(grid):
        R, C = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        # Flood fill from border with bg
        bg = 0
        visited = [[False]*C for _ in range(R)]
        q = []
        for r in range(R):
            for c in range(C):
                if (r==0 or r==R-1 or c==0 or c==C-1) and grid[r][c] == bg:
                    q.append((r,c))
                    visited[r][c] = True
        while q:
            r, c = q.pop()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc] == bg:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        # Fill unvisited bg cells
        for r in range(R):
            for c in range(C):
                if grid[r][c] == bg and not visited[r][c]:
                    # find adjacent non-bg color
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<R and 0<=nc<C and grid[nr][nc] != bg:
                            result[r][c] = grid[nr][nc]
                            break
        return result
    if test_func(fill_interior, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    bg = 0
    visited = [[False]*C for _ in range(R)]
    q = []
    for r in range(R):
        for c in range(C):
            if (r==0 or r==R-1 or c==0 or c==C-1) and grid[r][c] == bg:
                q.append((r,c))
                visited[r][c] = True
    while q:
        r, c = q.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc] == bg:
                visited[nr][nc] = True
                q.append((nr, nc))
    for r in range(R):
        for c in range(C):
            if grid[r][c] == bg and not visited[r][c]:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and grid[nr][nc] != bg:
                        result[r][c] = grid[nr][nc]
                        break
    return result
"""
        save_solution(task_id, code)
        return True
    return False

def try_count_to_size(task_id, examples):
    """Output is count of non-bg cells as a 1x1 or Nx1 grid."""
    def count_nonbg(grid):
        cnt = sum(1 for row in grid for v in row if v != 0)
        return [[cnt]]
    if test_func(count_nonbg, examples):
        code = """def transform(grid):
    cnt = sum(1 for row in grid for v in row if v != 0)
    return [[cnt]]
"""
        save_solution(task_id, code)
        return True
    return False

def try_crop_nonbg(task_id, examples):
    """Crop to bounding box of non-background."""
    def crop(grid):
        bg = grid[0][0]  # assume corner is bg
        R, C = len(grid), len(grid[0])
        rows_with = [r for r in range(R) if any(grid[r][c] != bg for c in range(C))]
        cols_with = [c for c in range(C) if any(grid[r][c] != bg for r in range(R))]
        if not rows_with or not cols_with:
            return grid
        r0, r1 = rows_with[0], rows_with[-1]+1
        c0, c1 = cols_with[0], cols_with[-1]+1
        return [grid[r][c0:c1] for r in range(r0, r1)]
    if test_func(crop, examples):
        code = """def transform(grid):
    bg = grid[0][0]
    R, C = len(grid), len(grid[0])
    rows_with = [r for r in range(R) if any(grid[r][c] != bg for c in range(C))]
    cols_with = [c for c in range(C) if any(grid[r][c] != bg for r in range(R))]
    if not rows_with or not cols_with:
        return grid
    r0, r1 = rows_with[0], rows_with[-1]+1
    c0, c1 = cols_with[0], cols_with[-1]+1
    return [grid[r][c0:c1] for r in range(r0, r1)]
"""
        save_solution(task_id, code)
        return True
    # Try with bg=0
    def crop0(grid):
        R, C = len(grid), len(grid[0])
        rows_with = [r for r in range(R) if any(grid[r][c] != 0 for c in range(C))]
        cols_with = [c for c in range(C) if any(grid[r][c] != 0 for r in range(R))]
        if not rows_with or not cols_with:
            return grid
        r0, r1 = rows_with[0], rows_with[-1]+1
        c0, c1 = cols_with[0], cols_with[-1]+1
        return [grid[r][c0:c1] for r in range(r0, r1)]
    if test_func(crop0, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    rows_with = [r for r in range(R) if any(grid[r][c] != 0 for c in range(C))]
    cols_with = [c for c in range(C) if any(grid[r][c] != 0 for r in range(R))]
    if not rows_with or not cols_with:
        return grid
    r0, r1 = rows_with[0], rows_with[-1]+1
    c0, c1 = cols_with[0], cols_with[-1]+1
    return [grid[r][c0:c1] for r in range(r0, r1)]
"""
        save_solution(task_id, code)
        return True
    return False

def try_scale_up(task_id, examples):
    """Scale up by factor k."""
    for k in [2, 3, 4, 5]:
        def make_scale(k=k):
            def f(grid):
                result = []
                for row in grid:
                    new_row = []
                    for v in row:
                        new_row.extend([v]*k)
                    for _ in range(k):
                        result.append(new_row[:])
                return result
            return f
        fn = make_scale(k)
        if test_func(fn, examples):
            code = f"""def transform(grid):
    k = {k}
    result = []
    for row in grid:
        new_row = []
        for v in row:
            new_row.extend([v]*k)
        for _ in range(k):
            result.append(new_row[:])
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_scale_down(task_id, examples):
    """Downsample by factor k."""
    for k in [2, 3, 4, 5]:
        def make_down(k=k):
            def f(grid):
                R, C = len(grid), len(grid[0])
                if R % k != 0 or C % k != 0:
                    return grid
                return [[grid[r*k][c*k] for c in range(C//k)] for r in range(R//k)]
            return f
        fn = make_down(k)
        if test_func(fn, examples):
            code = f"""def transform(grid):
    k = {k}
    R, C = len(grid), len(grid[0])
    return [[grid[r*k][c*k] for c in range(C//k)] for r in range(R//k)]
"""
            save_solution(task_id, code)
            return True
    return False

def try_boolean_ops(task_id, examples):
    """Try AND/OR/XOR between halves."""
    # Check if grid has two halves
    for ex in examples:
        R, C = len(ex['input']), len(ex['input'][0])
        # vertical split
        if R % 2 == 0:
            half = R // 2
            top = ex['input'][:half]
            bot = ex['input'][half:]
            # AND
            def hand(grid):
                R2, C2 = len(grid), len(grid[0])
                h = R2 // 2
                return [[1 if grid[r][c] != 0 and grid[r+h][c] != 0 else 0 
                         for c in range(C2)] for r in range(h)]
            # XOR  
            def hxor(grid):
                R2, C2 = len(grid), len(grid[0])
                h = R2 // 2
                return [[1 if (grid[r][c]!=0) != (grid[r+h][c]!=0) else 0 
                         for c in range(C2)] for r in range(h)]
    return False

def try_majority_color(task_id, examples):
    """Output is single cell with majority/most common color."""
    def most_common(grid):
        cnt = Counter(v for row in grid for v in row if v != 0)
        if not cnt:
            return [[0]]
        return [[cnt.most_common(1)[0][0]]]
    if test_func(most_common, examples):
        code = """from collections import Counter
def transform(grid):
    cnt = Counter(v for row in grid for v in row if v != 0)
    if not cnt:
        return [[0]]
    return [[cnt.most_common(1)[0][0]]]
"""
        save_solution(task_id, code)
        return True
    return False

def try_unique_color(task_id, examples):
    """Output is single cell with color that appears once."""
    def unique_color(grid):
        cnt = Counter(v for row in grid for v in row if v != 0)
        uniq = [k for k, v in cnt.items() if v == 1]
        if len(uniq) == 1:
            return [[uniq[0]]]
        return [[0]]
    if test_func(unique_color, examples):
        code = """from collections import Counter
def transform(grid):
    cnt = Counter(v for row in grid for v in row if v != 0)
    uniq = [k for k, v in cnt.items() if v == 1]
    if len(uniq) == 1:
        return [[uniq[0]]]
    return [[0]]
"""
        save_solution(task_id, code)
        return True
    return False

def try_count_color(task_id, examples):
    """Output is count of specific color as single number."""
    # Try each possible color
    for color in range(1, 10):
        def make_count(color=color):
            def f(grid):
                cnt = sum(1 for row in grid for v in row if v == color)
                return [[cnt]]
            return f
        fn = make_count(color)
        if test_func(fn, examples):
            code = f"""def transform(grid):
    cnt = sum(1 for row in grid for v in row if v == {color})
    return [[cnt]]
"""
            save_solution(task_id, code)
            return True
    return False

def try_mirror_half(task_id, examples):
    """Output is input with one half mirrored."""
    def mirror_right_from_left(grid):
        return [row[:len(row)//2] + row[:len(row)//2][::-1] for row in grid]
    def mirror_left_from_right(grid):
        return [row[len(row)//2:][::-1] + row[len(row)//2:] for row in grid]
    def mirror_bottom_from_top(grid):
        half = len(grid)//2
        return grid[:half] + grid[:half][::-1]
    def mirror_top_from_bottom(grid):
        half = len(grid)//2
        return grid[half:][::-1] + grid[half:]

    for fn, name in [(mirror_right_from_left, 'mirror_right_from_left'),
                     (mirror_left_from_right, 'mirror_left_from_right'),
                     (mirror_bottom_from_top, 'mirror_bottom_from_top'),
                     (mirror_top_from_bottom, 'mirror_top_from_bottom')]:
        if test_func(fn, examples):
            code = f"""def transform(grid):
    # {name}
"""
            if name == 'mirror_right_from_left':
                code += "    return [row[:len(row)//2] + row[:len(row)//2][::-1] for row in grid]\n"
            elif name == 'mirror_left_from_right':
                code += "    return [row[len(row)//2:][::-1] + row[len(row)//2:] for row in grid]\n"
            elif name == 'mirror_bottom_from_top':
                code += "    half = len(grid)//2\n    return grid[:half] + grid[:half][::-1]\n"
            elif name == 'mirror_top_from_bottom':
                code += "    half = len(grid)//2\n    return grid[half:][::-1] + grid[half:]\n"
            save_solution(task_id, code)
            return True
    return False

def try_replace_color(task_id, examples):
    """Replace one specific color with another."""
    for from_c in range(0, 10):
        for to_c in range(0, 10):
            if from_c == to_c:
                continue
            def make_replace(fc=from_c, tc=to_c):
                def f(grid):
                    return [[tc if v == fc else v for v in row] for row in grid]
                return f
            fn = make_replace(from_c, to_c)
            if test_func(fn, examples):
                code = f"""def transform(grid):
    return [[{to_c} if v == {from_c} else v for v in row] for row in grid]
"""
                save_solution(task_id, code)
                return True
    return False

def try_hollow(task_id, examples):
    """Make solid shapes hollow."""
    def hollow(grid):
        R, C = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(1, R-1):
            for c in range(1, C-1):
                if grid[r][c] != 0:
                    # Check if surrounded by same or non-bg
                    neighbors = [grid[r+dr][c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]]
                    if all(n != 0 for n in neighbors):
                        result[r][c] = 0
        return result
    if test_func(hollow, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(1, R-1):
        for c in range(1, C-1):
            if grid[r][c] != 0:
                neighbors = [grid[r+dr][c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]]
                if all(n != 0 for n in neighbors):
                    result[r][c] = 0
    return result
"""
        save_solution(task_id, code)
        return True
    return False

def try_sort_rows(task_id, examples):
    """Sort rows by various criteria."""
    def sort_rows_by_nonbg_count(grid):
        return sorted(grid, key=lambda row: sum(1 for v in row if v != 0))
    def sort_rows_by_nonbg_count_desc(grid):
        return sorted(grid, key=lambda row: sum(1 for v in row if v != 0), reverse=True)
    for fn, name in [(sort_rows_by_nonbg_count, 'asc'), (sort_rows_by_nonbg_count_desc, 'desc')]:
        if test_func(fn, examples):
            rev = 'True' if name == 'desc' else 'False'
            code = f"""def transform(grid):
    return sorted(grid, key=lambda row: sum(1 for v in row if v != 0), reverse={rev})
"""
            save_solution(task_id, code)
            return True
    return False

def try_border_fill(task_id, examples):
    """Fill border with color, or clear border."""
    # Detect most common border color in output
    def fill_border_color(color):
        def f(grid):
            R, C = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            for r in range(R):
                for c in range(C):
                    if r == 0 or r == R-1 or c == 0 or c == C-1:
                        result[r][c] = color
            return result
        return f
    for c in range(1, 10):
        if test_func(fill_border_color(c), examples):
            code = f"""def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(R):
        for c in range(C):
            if r == 0 or r == R-1 or c == 0 or c == C-1:
                result[r][c] = {c}
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_extract_pattern(task_id, examples):
    """Try to extract a sub-pattern (second half, quadrant, etc.)."""
    # Top-left quadrant
    def tl_quad(grid):
        R, C = len(grid), len(grid[0])
        return [grid[r][:C//2] for r in range(R//2)]
    def tr_quad(grid):
        R, C = len(grid), len(grid[0])
        return [grid[r][C//2:] for r in range(R//2)]
    def bl_quad(grid):
        R, C = len(grid), len(grid[0])
        return [grid[r][:C//2] for r in range(R//2, R)]
    def br_quad(grid):
        R, C = len(grid), len(grid[0])
        return [grid[r][C//2:] for r in range(R//2, R)]
    def left_half(grid):
        C = len(grid[0])
        return [row[:C//2] for row in grid]
    def right_half(grid):
        C = len(grid[0])
        return [row[C//2:] for row in grid]
    def top_half(grid):
        return grid[:len(grid)//2]
    def bottom_half(grid):
        return grid[len(grid)//2:]

    for fn, name in [(tl_quad,'tl'), (tr_quad,'tr'), (bl_quad,'bl'), (br_quad,'br'),
                     (left_half,'left'), (right_half,'right'), (top_half,'top'), (bottom_half,'bottom')]:
        if test_func(fn, examples):
            code = f"""def transform(grid):
    R, C = len(grid), len(grid[0])
"""
            if name == 'tl':
                code += "    return [grid[r][:C//2] for r in range(R//2)]\n"
            elif name == 'tr':
                code += "    return [grid[r][C//2:] for r in range(R//2)]\n"
            elif name == 'bl':
                code += "    return [grid[r][:C//2] for r in range(R//2, R)]\n"
            elif name == 'br':
                code += "    return [grid[r][C//2:] for r in range(R//2, R)]\n"
            elif name == 'left':
                code += "    return [row[:C//2] for row in grid]\n"
            elif name == 'right':
                code += "    return [row[C//2:] for row in grid]\n"
            elif name == 'top':
                code += "    return grid[:R//2]\n"
            elif name == 'bottom':
                code += "    return grid[R//2:]\n"
            save_solution(task_id, code)
            return True
    return False

def try_invert(task_id, examples):
    """Invert colors (swap bg with fg)."""
    def invert(grid):
        bg = 0
        # Find most common color
        cnt = Counter(v for row in grid for v in row)
        bg = cnt.most_common(1)[0][0]
        fg_colors = [k for k in cnt if k != bg]
        if len(fg_colors) == 1:
            fg = fg_colors[0]
            return [[fg if v == bg else bg for v in row] for row in grid]
        return [[0 if v != 0 else 1 for v in row] for row in grid]
    if test_func(invert, examples):
        code = """from collections import Counter
def transform(grid):
    cnt = Counter(v for row in grid for v in row)
    bg = cnt.most_common(1)[0][0]
    fg_colors = [k for k in cnt if k != bg]
    if len(fg_colors) == 1:
        fg = fg_colors[0]
        return [[fg if v == bg else bg for v in row] for row in grid]
    return [[0 if v != 0 else 1 for v in row] for row in grid]
"""
        save_solution(task_id, code)
        return True
    return False

def try_outline_cells(task_id, examples):
    """Outline non-bg cells (replace non-bg with border of its color, interior with bg)."""
    def outline_nonbg(grid):
        R, C = len(grid), len(grid[0])
        result = [[0]*C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if grid[r][c] != 0:
                    # Check if it's on the border of its component
                    is_border = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<R and 0<=nc<C) or grid[nr][nc] != grid[r][c]:
                            is_border = True
                            break
                    if is_border:
                        result[r][c] = grid[r][c]
        return result
    if test_func(outline_nonbg, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0:
                is_border = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0<=nr<R and 0<=nc<C) or grid[nr][nc] != grid[r][c]:
                        is_border = True
                        break
                if is_border:
                    result[r][c] = grid[r][c]
    return result
"""
        save_solution(task_id, code)
        return True
    return False

def try_connect_dots(task_id, examples):
    """Connect isolated points with lines."""
    def connect_h(grid):
        result = [row[:] for row in grid]
        R, C = len(grid), len(grid[0])
        for r in range(R):
            pts = [(c, grid[r][c]) for c in range(C) if grid[r][c] != 0]
            if len(pts) >= 2:
                c0, v = pts[0]
                c1, _ = pts[-1]
                for c in range(c0, c1+1):
                    if result[r][c] == 0:
                        result[r][c] = v
        return result
    def connect_v(grid):
        result = [row[:] for row in grid]
        R, C = len(grid), len(grid[0])
        for c in range(C):
            pts = [(r, grid[r][c]) for r in range(R) if grid[r][c] != 0]
            if len(pts) >= 2:
                r0, v = pts[0]
                r1, _ = pts[-1]
                for r in range(r0, r1+1):
                    if result[r][c] == 0:
                        result[r][c] = v
        return result
    for fn, name in [(connect_h, 'h'), (connect_v, 'v')]:
        if test_func(fn, examples):
            if name == 'h':
                code = """def transform(grid):
    result = [row[:] for row in grid]
    R, C = len(grid), len(grid[0])
    for r in range(R):
        pts = [(c, grid[r][c]) for c in range(C) if grid[r][c] != 0]
        if len(pts) >= 2:
            c0, v = pts[0]
            c1, _ = pts[-1]
            for c in range(c0, c1+1):
                if result[r][c] == 0:
                    result[r][c] = v
    return result
"""
            else:
                code = """def transform(grid):
    result = [row[:] for row in grid]
    R, C = len(grid), len(grid[0])
    for c in range(C):
        pts = [(r, grid[r][c]) for r in range(R) if grid[r][c] != 0]
        if len(pts) >= 2:
            r0, v = pts[0]
            r1, _ = pts[-1]
            for r in range(r0, r1+1):
                if result[r][c] == 0:
                    result[r][c] = v
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_xor_grids(task_id, examples):
    """XOR two halves of a grid."""
    def xor_halves_v(grid):
        R, C = len(grid), len(grid[0])
        if C % 2 != 0: return grid
        half = C // 2
        result = []
        for row in grid:
            new_row = []
            for c in range(half):
                a = row[c] != 0
                b = row[c + half] != 0
                new_row.append(row[c] if (a and not b) else (row[c+half] if (b and not a) else 0))
            result.append(new_row)
        return result
    def xor_halves_h(grid):
        R, C = len(grid), len(grid[0])
        if R % 2 != 0: return grid
        half = R // 2
        result = []
        for r in range(half):
            new_row = []
            for c in range(C):
                a = grid[r][c] != 0
                b = grid[r+half][c] != 0
                new_row.append(grid[r][c] if (a and not b) else (grid[r+half][c] if (b and not a) else 0))
            result.append(new_row)
        return result
    for fn, name in [(xor_halves_v, 'xor_v'), (xor_halves_h, 'xor_h')]:
        if test_func(fn, examples):
            if name == 'xor_v':
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = C // 2
    result = []
    for row in grid:
        new_row = []
        for c in range(half):
            a = row[c] != 0
            b = row[c + half] != 0
            new_row.append(row[c] if (a and not b) else (row[c+half] if (b and not a) else 0))
        result.append(new_row)
    return result
"""
            else:
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = R // 2
    result = []
    for r in range(half):
        new_row = []
        for c in range(C):
            a = grid[r][c] != 0
            b = grid[r+half][c] != 0
            new_row.append(grid[r][c] if (a and not b) else (grid[r+half][c] if (b and not a) else 0))
        result.append(new_row)
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_diagonal_mirror(task_id, examples):
    """Various diagonal operations."""
    def fill_diagonal(grid):
        R, C = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(R):
            for c in range(C):
                if result[r][c] == 0 and r < R and c < C:
                    if c < R and r < C:
                        result[r][c] = grid[c][r] if c < R and r < C else 0
        return result
    return False

def try_checkerboard(task_id, examples):
    """Fill with checkerboard pattern."""
    def checker(grid):
        return [[(r+c)%2 for c in range(len(grid[0]))] for r in range(len(grid))]
    if test_func(checker, examples):
        code = """def transform(grid):
    return [[(r+c)%2 for c in range(len(grid[0]))] for r in range(len(grid))]
"""
        save_solution(task_id, code)
        return True
    return False

def try_and_halves(task_id, examples):
    """AND two halves."""
    def and_h(grid):
        R, C = len(grid), len(grid[0])
        if C % 2 != 0: return grid
        half = C // 2
        result = []
        for row in grid:
            new_row = []
            for c in range(half):
                a = row[c] != 0
                b = row[c + half] != 0
                new_row.append(row[c] if (a and b) else 0)
            result.append(new_row)
        return result
    def and_v(grid):
        R, C = len(grid), len(grid[0])
        if R % 2 != 0: return grid
        half = R // 2
        result = []
        for r in range(half):
            new_row = []
            for c in range(C):
                a = grid[r][c] != 0
                b = grid[r+half][c] != 0
                new_row.append(grid[r][c] if (a and b) else 0)
            result.append(new_row)
        return result
    for fn, name in [(and_h, 'and_h'), (and_v, 'and_v')]:
        if test_func(fn, examples):
            if name == 'and_h':
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = C // 2
    result = []
    for row in grid:
        new_row = []
        for c in range(half):
            a = row[c] != 0
            b = row[c + half] != 0
            new_row.append(row[c] if (a and b) else 0)
        result.append(new_row)
    return result
"""
            else:
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = R // 2
    result = []
    for r in range(half):
        new_row = []
        for c in range(C):
            a = grid[r][c] != 0
            b = grid[r+half][c] != 0
            new_row.append(grid[r][c] if (a and b) else 0)
        result.append(new_row)
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_or_halves(task_id, examples):
    """OR two halves."""
    def or_h(grid):
        R, C = len(grid), len(grid[0])
        if C % 2 != 0: return grid
        half = C // 2
        result = []
        for row in grid:
            new_row = []
            for c in range(half):
                new_row.append(row[c] if row[c] != 0 else row[c + half])
            result.append(new_row)
        return result
    def or_v(grid):
        R, C = len(grid), len(grid[0])
        if R % 2 != 0: return grid
        half = R // 2
        result = []
        for r in range(half):
            new_row = []
            for c in range(C):
                new_row.append(grid[r][c] if grid[r][c] != 0 else grid[r+half][c])
            result.append(new_row)
        return result
    for fn, name in [(or_h, 'or_h'), (or_v, 'or_v')]:
        if test_func(fn, examples):
            if name == 'or_h':
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = C // 2
    result = []
    for row in grid:
        new_row = []
        for c in range(half):
            new_row.append(row[c] if row[c] != 0 else row[c + half])
        result.append(new_row)
    return result
"""
            else:
                code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    half = R // 2
    result = []
    for r in range(half):
        new_row = []
        for c in range(C):
            new_row.append(grid[r][c] if grid[r][c] != 0 else grid[r+half][c])
        result.append(new_row)
    return result
"""
            save_solution(task_id, code)
            return True
    return False

def try_row_col_count(task_id, examples):
    """Output reflects counts of colors per row or column."""
    # Count non-bg per row as output
    def row_counts(grid):
        return [[sum(1 for v in row if v != 0)] for row in grid]
    if test_func(row_counts, examples):
        code = """def transform(grid):
    return [[sum(1 for v in row if v != 0)] for row in grid]
"""
        save_solution(task_id, code)
        return True
    def col_counts(grid):
        R, C = len(grid), len(grid[0])
        return [[sum(1 for r in range(R) if grid[r][c] != 0) for c in range(C)]]
    if test_func(col_counts, examples):
        code = """def transform(grid):
    R, C = len(grid), len(grid[0])
    return [[sum(1 for r in range(R) if grid[r][c] != 0) for c in range(C)]]
"""
        save_solution(task_id, code)
        return True
    return False

def try_all(task_id, examples):
    solvers = [
        try_simple,
        try_color_map,
        try_tiling,
        try_scale_up,
        try_scale_down,
        try_gravity,
        try_crop_nonbg,
        try_extract_pattern,
        try_mirror_half,
        try_replace_color,
        try_invert,
        try_outline_cells,
        try_fill_interior_wrapper,
        try_connect_dots,
        try_xor_grids,
        try_and_halves,
        try_or_halves,
        try_majority_color,
        try_unique_color,
        try_count_color,
        try_count_to_size,
        try_hollow,
        try_sort_rows,
        try_border_fill,
        try_row_col_count,
        try_checkerboard,
    ]
    for solver in solvers:
        try:
            if solver(task_id, examples):
                return True
        except:
            pass
    return False

def try_fill_interior_wrapper(task_id, examples):
    return try_outline(task_id, examples)

def main():
    with open(BATCH_FILE) as f:
        task_ids = json.load(f)
    
    solved = 0
    failed = 0
    already = 0
    
    for i, task_id in enumerate(task_ids):
        out_path = os.path.join(OUT_DIR, f"{task_id}.py")
        if os.path.exists(out_path):
            already += 1
            continue
        
        try:
            task = load_task(task_id)
        except FileNotFoundError:
            print(f"  MISSING: {task_id}")
            failed += 1
            continue
        
        train = task['train']
        print(f"[{i+1}/{len(task_ids)}] {task_id} ({len(train)} train examples)", end="")
        
        if try_all(task_id, train):
            solved += 1
            print(f" -> SOLVED")
        else:
            failed += 1
            print(f" -> FAILED")
    
    print(f"\nDone: {solved} solved, {failed} failed, {already} already done")

if __name__ == '__main__':
    main()
