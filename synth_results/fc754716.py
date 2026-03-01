
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    # Find center color
    color = 0
    for r in grid:
        for v in r:
            if v != 0:
                color = v
                break
    out = [[0]*cols for _ in range(rows)]
    for c in range(cols):
        out[0][c] = color
        out[rows-1][c] = color
    for r in range(rows):
        out[r][0] = color
        out[r][cols-1] = color
    return out
