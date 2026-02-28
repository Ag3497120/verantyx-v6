import numpy as np
def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    pos2 = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 2][0]
    r2, c2 = pos2
    new_g = np.delete(g, c2, axis=1)
    rot = (H - r2) % H
    return np.roll(new_g, -rot, axis=0).tolist()
