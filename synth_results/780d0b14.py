def transform(grid):
    return _solve(grid)

def solve_780d0b14(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find 0-rows and 0-columns that separate sections
    zero_rows = [r for r in range(H) if np.all(g[r,:] == 0)]
    zero_cols = [c for c in range(W) if np.all(g[:,c] == 0)]
    
    # Find row groups and col groups
    row_groups = []
    prev = -1
    for r in sorted(zero_rows + [H]):
        if r > prev+1:
            row_groups.append((prev+1, r))
        prev = r
    if not row_groups: row_groups = [(0, H)]
    
    col_groups = []
    prev = -1
    for c in sorted(zero_cols + [W]):
        if c > prev+1:
            col_groups.append((prev+1, c))
        prev = c
    if not col_groups: col_groups = [(0, W)]
    
    result = []
    for (r0, r1) in row_groups:
        row_result = []
        for (c0, c1) in col_groups:
            region = g[r0:r1, c0:c1]
            vals = [v for v in region.flatten() if v != 0]
            if vals:
                color = Counter(vals).most_common(1)[0][0]
                row_result.append(int(color))
            # else skip
        if row_result:
            result.append(row_result)
    return result


_solve = solve_780d0b14
