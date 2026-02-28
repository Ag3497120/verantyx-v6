import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    struct = np.ones((3, 3), int)
    labeled, n = label(g != 0, structure=struct)
    shapes = []
    for lbl in range(1, n + 1):
        mask = labeled == lbl
        rows, cols = np.where(mask)
        sub = g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()
        shapes.append((rows.min(), cols.min(), sub))
    
    all_rows = [s[0] for s in shapes]
    all_cols = [s[1] for s in shapes]
    max_r = max(s[0] + s[2].shape[0] for s in shapes)
    max_c = max(s[1] + s[2].shape[1] for s in shapes)
    min_r = min(all_rows)
    min_c = min(all_cols)
    row_span = max_r - min_r
    col_span = max_c - min_c
    
    if row_span >= col_span:
        shapes.sort(key=lambda s: s[0])
        result = np.concatenate([s[2] for s in shapes], axis=0)
    else:
        shapes.sort(key=lambda s: s[1])
        result = np.concatenate([s[2] for s in shapes], axis=1)
    
    return result.tolist()
