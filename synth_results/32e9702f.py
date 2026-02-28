def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.full_like(g, 5)  # background becomes 5
    
    for r in range(rows):
        row = g[r]
        # Find non-zero cells
        nz = [(c, row[c]) for c in range(cols) if row[c] != 0]
        # Shift each LEFT by 1 (clip at 0)
        for c, v in nz:
            nc = c - 1
            if nc >= 0:
                result[r, nc] = v
    
    return result.tolist()
