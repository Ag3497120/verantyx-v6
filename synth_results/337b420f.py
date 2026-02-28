def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find divider columns (all 0 or background)
    # Panels separated by columns of 0
    bg = 0
    div_cols = [c for c in range(cols) if all(g[r, c] == bg for r in range(rows))]
    
    if not div_cols:
        return grid
    
    # Find panels
    panels = []
    prev = 0
    for dc in div_cols:
        if dc > prev:
            panels.append(g[:, prev:dc])
        prev = dc + 1
    if prev < cols:
        panels.append(g[:, prev:])
    
    if not panels:
        return grid
    
    # Each panel should have the same number of rows
    # Merge: take one row from each panel and stack
    # Output: rows from panels interleaved
    
    # Find non-zero rows in each panel
    result_rows = []
    for p in panels:
        for r in range(p.shape[0]):
            if any(p[r, c] != bg for c in range(p.shape[1])):
                result_rows.append(list(p[r]))
    
    return result_rows if result_rows else grid.tolist()
