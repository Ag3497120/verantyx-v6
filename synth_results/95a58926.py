def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    R, C = g.shape
    line_color = 5
    
    divider_rows = [r for r in range(R) if (g[r] != 0).sum() > C * 0.7]
    divider_cols = [c for c in range(C) if (g[:, c] != 0).sum() > R * 0.7]
    
    # Find mark color: most common non-5 color on divider lines
    line_cells = set()
    for r in divider_rows:
        for c in range(C): line_cells.add((r,c))
    for c in divider_cols:
        for r in range(R): line_cells.add((r,c))
    
    mark_counts = Counter()
    for r, c in line_cells:
        v = int(g[r, c])
        if v != 0 and v != line_color:
            mark_counts[v] += 1
    
    mark_color = mark_counts.most_common(1)[0][0] if mark_counts else 3
    
    out = np.zeros_like(g)
    for r in divider_rows:
        out[r, :] = line_color
    for c in divider_cols:
        out[:, c] = line_color
    for r in divider_rows:
        for c in divider_cols:
            out[r, c] = mark_color
    
    return out.tolist()
