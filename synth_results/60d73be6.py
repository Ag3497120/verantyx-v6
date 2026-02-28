import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find background (most common)
    cnt = Counter(v for v in g.flat)
    bg = cnt.most_common(1)[0][0]
    
    # Find the cross color (fills an entire row and column)
    cross_color = None
    cross_row = cross_col = None
    for r in range(H):
        if all(g[r, c] != bg for c in range(W)):
            cross_row = r
            cross_color = g[r, 0]
            break
    for c in range(W):
        if all(g[r, c] != bg for r in range(H)):
            cross_col = c
            break
    
    # Find non-bg, non-cross values and their quadrant
    non_bg = {}
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v != bg and v != cross_color:
                non_bg[(r, c)] = v
    
    out = np.copy(g)
    
    # Reflect across cross_row and cross_col
    for (r, c), v in non_bg.items():
        # Original position
        out[r, c] = v
        # Reflect across vertical axis (cross_col)
        rc = 2 * cross_col - c
        if 0 <= rc < W and rc != cross_col:
            out[r, rc] = v
        # Reflect across horizontal axis (cross_row)
        rr = 2 * cross_row - r
        if 0 <= rr < H and rr != cross_row:
            out[rr, c] = v
        # Reflect across both
        if 0 <= rr < H and 0 <= rc < W and rr != cross_row and rc != cross_col:
            out[rr, rc] = v
    
    return out.tolist()
