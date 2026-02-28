import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    out = np.copy(g)
    H, W = g.shape
    
    # Use 4-connectivity
    struct4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(g == 1, structure=struct4)
    
    for lbl in range(1, n + 1):
        mask = labeled == lbl
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        
        # Check if rectangle outline (ring)
        is_ring = (h >= 3 and w >= 3)
        if is_ring:
            for r in range(r0, r1 + 1):
                if g[r, c0] != 1 or g[r, c1] != 1:
                    is_ring = False; break
        if is_ring:
            for c in range(c0, c1 + 1):
                if g[r0, c] != 1 or g[r1, c] != 1:
                    is_ring = False; break
        if is_ring:
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if g[r, c] != 0:
                        is_ring = False; break
                if not is_ring: break
        
        if is_ring:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if g[r, c] == 1:
                        out[r, c] = 0
            mid_r = (r0 + r1) / 2
            mid_c = (c0 + c1) / 2
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if abs(r - mid_r) < 0.6 or abs(c - mid_c) < 0.6:
                        out[r, c] = 2
    
    return out.tolist()
