def transform(grid):
    import copy
    g = [list(row) for row in grid]
    for r in range(len(g)):
        for c in range(0, len(g[r]), 3):
            if g[r][c] == 4:
                g[r][c] = 6
    return g
