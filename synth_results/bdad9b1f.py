def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    
    # Find vertical bar (column with 8s)
    vert_col = None
    for c in range(cols):
        col_vals = [g[r][c] for r in range(rows)]
        if any(v == 8 for v in col_vals):
            vert_col = c
            break
    
    # Find horizontal bar (row with 2s)
    horiz_row = None
    for r in range(rows):
        if any(g[r][c] == 2 for c in range(cols)):
            horiz_row = r
            break
    
    if vert_col is None or horiz_row is None:
        return grid
    
    # Extend vertical bar to full height
    for r in range(rows):
        if g[r][vert_col] == 0:
            g[r][vert_col] = 8
    
    # Extend horizontal bar to full width
    for c in range(cols):
        if g[horiz_row][c] == 0:
            g[horiz_row][c] = 2
    
    # Mark intersection with 4
    g[horiz_row][vert_col] = 4
    
    return g
