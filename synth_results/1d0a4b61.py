def transform(grid):
    import numpy as np
    
    g = np.array(grid, dtype=int)
    h, w = g.shape
    nonzero = (g != 0)
    
    # Find smallest repeating tile
    best_period = None
    for bh in range(1, h+1):
        for bw in range(1, w+1):
            valid = True
            for i in range(h):
                for j in range(w):
                    if not nonzero[i, j]:
                        continue
                    ri, rj = i % bh, j % bw
                    if not nonzero[ri, rj]:
                        continue
                    if g[i, j] != g[ri, rj]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_period = (bh, bw)
                break
        if best_period:
            break
    
    if not best_period:
        return grid
    
    bh, bw = best_period
    # Build tile from non-zero values in first tile
    tile = np.zeros((bh, bw), dtype=int)
    
    # Fill tile with known values
    for i in range(h):
        for j in range(w):
            if nonzero[i, j]:
                tile[i % bh, j % bw] = g[i, j]
    
    # Fill result using tile
    result = np.zeros_like(g)
    for i in range(h):
        for j in range(w):
            result[i, j] = tile[i % bh, j % bw]
    
    return result.tolist()
