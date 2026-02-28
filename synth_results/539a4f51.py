
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    # Determine n (period)
    last_row = g[-1]
    last_col = g[:, -1]
    if len(set(last_row.tolist())) == 1 and last_row[0] == last_col[-1]:
        border_val = int(last_row[0])
        if border_val == 0:
            n = h - 1  # bg border, pattern is inner (h-1)x(h-1)
        else:
            n = h  # border is outermost ring of pattern
    else:
        n = h
    # Extract diagonal color map from M (nxn top-left region)
    M_color = [int(g[d, d]) for d in range(n)]
    # Output = 2*h x 2*w
    oh, ow = 2*h, 2*w
    out = [[M_color[max(r,c) % n] for c in range(ow)] for r in range(oh)]
    return out
