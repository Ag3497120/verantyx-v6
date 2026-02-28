def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    new_C = 2 * C
    out = np.zeros((R, new_C), dtype=int)
    
    for r in range(R):
        row = g[r]
        if np.all(row == 0):
            continue
        # Find period by looking for smallest repeating unit
        period = 1
        for p in range(1, C+1):
            if np.all(row == np.tile(row[:p], C//p + 1)[:C]):
                period = p
                break
        # Tile row to new_C
        tiled = np.tile(row[:period], new_C // period + 1)[:new_C]
        out[r] = tiled
    
    return out.tolist()
