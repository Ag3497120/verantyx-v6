import numpy as np

def transform(grid):
    g = np.array(grid)
    h, w = g.shape
    bg = 7  # background is 7
    out = np.zeros_like(g)
    
    # Find non-7 values and their Chebyshev distances from center
    cx, cy = h//2, w//2
    
    # Find all non-bg values and map distance->color
    dist_color = {}
    for r in range(h):
        for c in range(w):
            if g[r,c] != bg:
                d = max(abs(r-cx), abs(c-cy))
                dist_color[d] = g[r,c]
    
    # Fill concentric rings
    for r in range(h):
        for c in range(w):
            d = max(abs(r-cx), abs(c-cy))
            if d in dist_color:
                out[r,c] = dist_color[d]
            else:
                out[r,c] = bg
    
    return out.tolist()
