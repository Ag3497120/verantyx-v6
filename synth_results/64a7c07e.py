import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    result = np.zeros_like(g)
    struct = np.ones((3, 3), int)  # 8-connectivity
    labeled, n = label(g != 0, structure=struct)
    for lbl in range(1, n + 1):
        mask = labeled == lbl
        rows, cols = np.where(mask)
        width = cols.max() - cols.min() + 1
        for r, c in zip(rows, cols):
            nc = c + width
            if 0 <= nc < W:
                result[r, nc] = g[r, c]
    return result.tolist()
