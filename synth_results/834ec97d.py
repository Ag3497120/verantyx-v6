def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    # Find non-zero cell
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return grid
    r, c = nz[0]
    val = g[r, c]
    out = np.zeros_like(g)
    # Fill rows 0..r with 4s at same-parity cols as c
    for row in range(r + 1):
        for col in range(W):
            if col % 2 == c % 2:
                out[row, col] = 4
    # Move non-zero to r+1
    if r + 1 < H:
        out[r + 1, c] = val
    return out.tolist()
