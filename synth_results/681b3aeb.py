import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    # Find two colored shapes
    colors = [v for v in np.unique(g) if v != 0]
    shapes = []
    for c in colors:
        mask = g == c
        rows, cols = np.where(mask)
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        sub = g[r1:r2+1, c1:c2+1].copy()
        sub[sub != c] = 0
        shapes.append((c, sub))
    
    # Try all offsets to combine the two shapes without conflict
    c1, s1 = shapes[0]
    c2, s2 = shapes[1]
    h1, w1 = s1.shape
    h2, w2 = s2.shape
    
    best = None
    best_size = float('inf')
    
    for dr in range(-(h2-1), h1):
        for dc in range(-(w2-1), w1):
            # Place s2 at offset (dr, dc) relative to s1
            min_r = min(0, dr); max_r = max(h1-1, dr+h2-1)
            min_c = min(0, dc); max_c = max(w1-1, dc+w2-1)
            H = max_r - min_r + 1
            W = max_c - min_c + 1
            canvas = np.zeros((H, W), int)
            # Place s1
            r1o, c1o = -min_r, -min_c
            canvas[r1o:r1o+h1, c1o:c1o+w1] += s1
            # Check s2 placement
            r2o, c2o = -min_r+dr, -min_c+dc
            conflict = False
            for rr in range(h2):
                for cc in range(w2):
                    if s2[rr, cc] != 0 and canvas[r2o+rr, c2o+cc] != 0:
                        conflict = True; break
                if conflict: break
            if not conflict:
                canvas[r2o:r2o+h2, c2o:c2o+w2] += s2
                size = H * W
                if size < best_size:
                    best_size = size
                    best = canvas.tolist()
    
    return best if best is not None else grid
