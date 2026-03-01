from collections import defaultdict

def transform(grid):
    rows = len(grid); cols = len(grid[0])
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r,c))
    result = [row[:] for row in grid]
    colors = list(color_cells.keys())
    if len(colors) < 2: return result
    bboxes = {}
    for color, cells in color_cells.items():
        minr = min(r for r,c in cells); maxr = max(r for r,c in cells)
        minc = min(c for r,c in cells); maxc = max(c for r,c in cells)
        bboxes[color] = (minr, maxr, minc, maxc)
    c1, c2 = colors[0], colors[1]
    r1_min,r1_max,c1_min,c1_max = bboxes[c1]
    r2_min,r2_max,c2_min,c2_max = bboxes[c2]
    if c1_max < c2_min - 1:
        r_over_min = max(r1_min, r2_min) + 1; r_over_max = min(r1_max, r2_max) - 1
        for r in range(r_over_min, r_over_max+1):
            for c in range(c1_max+1, c2_min):
                result[r][c] = 8
    elif c2_max < c1_min - 1:
        r_over_min = max(r1_min, r2_min) + 1; r_over_max = min(r1_max, r2_max) - 1
        for r in range(r_over_min, r_over_max+1):
            for c in range(c2_max+1, c1_min):
                result[r][c] = 8
    elif r1_max < r2_min - 1:
        c_over_min = max(c1_min, c2_min) + 1; c_over_max = min(c1_max, c2_max) - 1
        for r in range(r1_max+1, r2_min):
            for c in range(c_over_min, c_over_max+1):
                result[r][c] = 8
    elif r2_max < r1_min - 1:
        c_over_min = max(c1_min, c2_min) + 1; c_over_max = min(c1_max, c2_max) - 1
        for r in range(r2_max+1, r1_min):
            for c in range(c_over_min, c_over_max+1):
                result[r][c] = 8
    return result
