import numpy as np

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    zeros_set = set(map(tuple, np.argwhere(g == 0).tolist()))
    if not zeros_set:
        return grid
    
    rows = [r for r,c in zeros_set]
    r_min = min(rows)
    top_zeros = [(r,c) for r,c in zeros_set if r == r_min]
    if not top_zeros:
        return grid
    
    c_left = min(c for r,c in top_zeros)
    c_right = max(c for r,c in top_zeros)
    
    def trace_diag(r0, c0, dr, dc):
        path = []
        r, c = r0, c0
        while 0 <= r < h and 0 <= c < w:
            if (r, c) in zeros_set:
                path.append((r, c))
            r += dr
            c += dc
        return path
    
    # Try 4 combinations: (left, down-right), (right, down-left), (left, down-left), (right, down-right)
    candidates = [
        trace_diag(r_min, c_left, 1, 1),
        trace_diag(r_min, c_right, 1, -1),
        trace_diag(r_min, c_left, 1, -1),
        trace_diag(r_min, c_right, 1, 1),
    ]
    
    best = max(candidates, key=len)
    
    for r, c in best:
        out[r, c] = 8
    
    return out.tolist()
