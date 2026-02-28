
def transform(grid):
    import numpy as np
    g = np.array(grid)
    result = g.copy()
    rows, cols = g.shape
    for r in range(rows):
        nz = [c for c in range(cols) if g[r,c] != 0]
        if len(nz) >= 2:
            color = g[r, nz[0]]
            result[r, nz[0]:nz[-1]+1] = color
    for c in range(cols):
        nz = [r for r in range(rows) if g[r,c] != 0]
        if len(nz) >= 2:
            color = g[nz[0], c]
            result[nz[0]:nz[-1]+1, c] = color
    return result.tolist()
