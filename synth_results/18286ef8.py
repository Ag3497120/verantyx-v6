def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    zero_rows = [r for r in range(rows) if all(v==0 for v in grid[r])]
    zero_cols = [c for c in range(cols) if all(grid[r][c]==0 for r in range(rows))]
    center_r, center_c = None, None
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] == 9:
                neighbors = [grid[r+dr][c+dc] for dr in [-1,0,1] for dc in [-1,0,1] if not(dr==0 and dc==0)]
                if all(v==5 for v in neighbors):
                    center_r, center_c = r, c
    if center_r is None: return result
    six_r, six_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6:
                six_r, six_c = r, c
    if six_r is None: return result
    def get_row_band(r):
        return sum(1 for zr in zero_rows if r > zr)
    def get_col_band(c):
        return sum(1 for zc in zero_cols if c > zc)
    cb_c = get_row_band(center_r); cb_s = get_row_band(six_r)
    cc_c = get_col_band(center_c); cc_s = get_col_band(six_c)
    dr = 0 if cb_c == cb_s else (1 if six_r > center_r else -1)
    dc = 0 if cc_c == cc_s else (1 if six_c > center_c else -1)
    result[center_r][center_c] = 5
    result[center_r+dr][center_c+dc] = 9
    result[six_r][six_c] = 9
    return result
