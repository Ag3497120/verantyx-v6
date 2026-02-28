def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    bg = 0
    # Find pattern (non-zero bounding box)
    nz = np.argwhere(g != bg)
    if len(nz) == 0: return grid
    r1, r2 = nz[:,0].min(), nz[:,0].max()
    c1, c2 = nz[:,1].min(), nz[:,1].max()
    pattern = g[r1:r2+1, c1:c2+1]
    pH, pW = pattern.shape
    # Output is H x W (same as input)
    # Grid is divided into pH x pW tiles
    out = np.zeros_like(g)
    # For each non-zero cell (i,j) in pattern, place pattern at tile (i,j)
    for i in range(pH):
        for j in range(pW):
            if pattern[i, j] != bg:
                tr = i * pH
                tc = j * pW
                if tr + pH <= H and tc + pW <= W:
                    out[tr:tr+pH, tc:tc+pW] = pattern
    return out.tolist()
