def transform(grid):
    palette = grid[0]
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    idx = 0
    for r in range(2, rows):
        color = palette[idx % len(palette)]
        g[r] = [color] * cols
        idx += 1
    return g
