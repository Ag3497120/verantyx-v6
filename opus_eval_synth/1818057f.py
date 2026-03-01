def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid[r][c] == 4 and grid[r-1][c] == 4 and grid[r+1][c] == 4
                    and grid[r][c-1] == 4 and grid[r][c+1] == 4):
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    out[r + dr][c + dc] = 8
    return out
