import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find background (most common)
    from collections import Counter
    bg = Counter(v for v in g.flat).most_common(1)[0][0]
    
    mask = g != bg
    struct4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(mask, structure=struct4)
    
    out = np.copy(g)
    
    for lbl in range(1, n + 1):
        cc = labeled == lbl
        size = int(cc.sum())
        
        if size <= 3:
            continue  # small comps stay
        
        rows, cols = np.where(cc)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        oh = r1 - r0 + 1
        
        # Extract shape
        sub = g[r0:r1+1, c0:c1+1].copy()
        
        # Clear original
        for r, c in zip(rows.tolist(), cols.tolist()):
            out[r, c] = bg
        
        # Place at bottom
        new_r0 = H - oh
        new_r1 = H - 1
        for dr in range(oh):
            for dc in range(c1 - c0 + 1):
                if sub[dr, dc] != bg:
                    out[new_r0 + dr, c0 + dc] = int(sub[dr, dc])
    
    return out.tolist()
