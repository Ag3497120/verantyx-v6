def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    for c in range(cols):
        if g[0, c] != 0:
            color = g[0, c]
            parity = c % 2  # parity of the signal cell (0+c)
            for dc in range(-1, 2):
                nc = c + dc
                if 0 <= nc < cols:
                    for r in range(rows):
                        if (r + nc) % 2 == parity:
                            result[r, nc] = color
    
    return result.tolist()
