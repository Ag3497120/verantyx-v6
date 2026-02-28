import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    # Background = most common value
    flat = g.flatten()
    bg = Counter(flat.tolist()).most_common(1)[0][0]
    
    # Find 0-cells region (the hole to fill)
    zero_mask = (g == 0)
    zero_rows = np.where(zero_mask.any(axis=1))[0]
    zero_cols = np.where(zero_mask.any(axis=0))[0]
    if len(zero_rows) == 0:
        return out.tolist()
    zr1, zr2 = zero_rows[0], zero_rows[-1]
    zc1, zc2 = zero_cols[0], zero_cols[-1]
    
    # Find pattern cells (non-bg, non-zero)
    pat_mask = (g != bg) & (g != 0)
    pat_rows = np.where(pat_mask.any(axis=1))[0]
    pat_cols = np.where(pat_mask.any(axis=0))[0]
    if len(pat_rows) == 0:
        return out.tolist()
    pr1, pr2 = pat_rows[0], pat_rows[-1]
    pc1, pc2 = pat_cols[0], pat_cols[-1]
    
    # Extract pattern subgrid
    pattern = g[pr1:pr2+1, pc1:pc2+1]
    
    # Flip horizontally and place in zero region
    flipped = pattern[:, ::-1]
    
    out[zr1:zr2+1, zc1:zc2+1] = flipped
    
    return out.tolist()
