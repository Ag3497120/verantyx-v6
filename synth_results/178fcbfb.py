def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    # Find value-2 cells (vertical lines) and other non-zero cells (horizontal lines)
    v_lines = []  # (col, value=2) - will draw vertical
    h_lines = []  # (row, value) - will draw horizontal
    
    for r in range(rows):
        for c in range(cols):
            v = g[r, c]
            if v == 0:
                continue
            if v == 2:
                v_lines.append((r, c, v))
            else:
                h_lines.append((r, c, v))
    
    # Draw vertical lines first (2s)
    for r, c, v in v_lines:
        result[:, c] = v
    
    # Draw horizontal lines second (override)
    for r, c, v in h_lines:
        result[r, :] = v
    
    return result.tolist()
