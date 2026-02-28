def transform(grid):
    rows, cols = len(grid), len(grid[0])
    g = [list(row) for row in grid]
    zero_cols = set(c for c in range(cols) if all(grid[r][c]==0 for r in range(rows)))
    zero_rows = set(r for r in range(rows) if all(grid[r][c]==0 for c in range(cols)))
    for c in zero_cols:
        for r in range(rows): g[r][c] = 2
    for r in zero_rows:
        for c in range(cols): g[r][c] = 2
    return g
