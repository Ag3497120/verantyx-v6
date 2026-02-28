def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    
    for r in range(rows):
        left_color = g[r][0]
        for c in range(cols):
            if g[r][c] == 5:
                g[r][c] = left_color
    return g
