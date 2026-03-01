from math import gcd
from functools import reduce
from collections import Counter

def transform(grid):
    def find_period(positions):
        if len(positions) < 2: return None
        positions = sorted(positions)
        spacings = [positions[i+1]-positions[i] for i in range(len(positions)-1)]
        return reduce(gcd, spacings)

    def extend_line(cells, length):
        if not cells: return {}
        val_counts = Counter(v for _,v in cells)
        repeating = {v: sorted(p for p,vv in cells if vv==v) for v,c in val_counts.items() if c >= 2}
        single = {v: [p for p,vv in cells if vv==v][0] for v,c in val_counts.items() if c == 1}
        if len(repeating) != 1: return {}
        if len(single) > 1: return {}
        rep_val, positions = list(repeating.items())[0]
        period = find_period(positions)
        if period is None: return {}
        base = positions[0] % period
        true_color = None
        true_pos = None
        if single:
            sv, sp = list(single.items())[0]
            if sp % period == base:
                true_color = sv
                true_pos = sp
        fill_color = true_color if true_color else rep_val
        if true_color:
            min_pos = min(min(positions), true_pos)
            max_pos = max(max(positions), true_pos)
        else:
            min_pos = 0
            max_pos = length - 1
        result = {}
        for i in range(length):
            if i % period == base and min_pos <= i <= max_pos:
                result[i] = fill_color
        return result

    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(R):
        cells = [(c, grid[r][c]) for c in range(C) if grid[r][c] != 0]
        for pos, val in extend_line(cells, C).items():
            result[r][pos] = val
    for c in range(C):
        cells = [(r, grid[r][c]) for r in range(R) if grid[r][c] != 0]
        for pos, val in extend_line(cells, R).items():
            result[pos][c] = val
    return result
