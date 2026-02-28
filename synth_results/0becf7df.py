def transform(grid):
    import numpy as np
    g = np.array(grid)
    
    # Key is at (0,0)-(1,1): defines color swaps
    # key[0,0] <-> key[0,1] and key[1,0] <-> key[1,1]
    k00, k01 = g[0,0], g[0,1]
    k10, k11 = g[1,0], g[1,1]
    
    mapping = {}
    if k00 != 0 and k01 != 0:
        mapping[k00] = k01
        mapping[k01] = k00
    if k10 != 0 and k11 != 0:
        mapping[k10] = k11
        mapping[k11] = k10
    
    result = g.copy()
    rows, cols = g.shape
    for r in range(rows):
        for c in range(cols):
            # Don't transform the key itself
            if r <= 1 and c <= 1:
                continue
            v = g[r, c]
            if v in mapping:
                result[r, c] = mapping[v]
    
    return result.tolist()
