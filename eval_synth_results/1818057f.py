def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    centers = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] == 4 and grid[r-1][c] == 4 and grid[r+1][c] == 4 and grid[r][c-1] == 4 and grid[r][c+1] == 4:
                centers.append((r, c))
    for r, c in centers:
        out[r][c] = 8
        out[r-1][c] = 8
        out[r+1][c] = 8
        out[r][c-1] = 8
        out[r][c+1] = 8
    return out
