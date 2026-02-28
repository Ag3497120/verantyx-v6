def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    bar = g[0].tolist()
    wall_color = int(g[1, 0])
    
    isolated = None
    for r in range(2, R):
        for c in range(C):
            if g[r, c] != 0:
                isolated = (r, c, int(g[r, c]))
                break
        if isolated: break
    
    if isolated is None:
        return grid
    
    ir, ic, iv = isolated
    out = g.copy()
    
    for r in range(2, R):
        for c in range(C):
            d = max(abs(r - ir), abs(c - ic))
            if d < len(bar) and bar[d] != 0:
                out[r, c] = bar[d]
    
    # Replace bar entry at isolated cell's col with wall_color (if non-zero)
    if bar[ic] != 0:
        out[0, ic] = wall_color
    
    return out.tolist()
