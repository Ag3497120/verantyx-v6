
def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    rows, cols = g.shape
    # Detect background (most common value)
    cnt = Counter(g.flatten().tolist())
    bg = cnt.most_common(1)[0][0]
    # Find non-bg bounding box
    nz = np.where(g != bg)
    if len(nz[0]) == 0:
        return grid
    r0 = nz[0].min(); r1 = nz[0].max()
    c0 = nz[1].min(); c1 = nz[1].max()
    pattern = g[r0:r1+1, c0:c1+1]
    h, w = pattern.shape
    
    # Detect border: check if bottom row and right col are same color (border)
    border_color = None
    bottom_row = pattern[h-1, :]
    right_col = pattern[:, w-1]
    if len(set(bottom_row.tolist())) == 1 and len(set(right_col.tolist())) == 1:
        bc1 = bottom_row[0]
        bc2 = right_col[0]
        if bc1 == bc2 and bc1 != bg:
            border_color = bc1
    
    if border_color is not None:
        core = pattern[:h-1, :w-1]
        core_h, core_w = core.shape
    else:
        core = pattern
        core_h, core_w = h, w
        border_color = None
    
    # Build sequence
    core_row0 = list(core[0, :])
    if border_color is not None:
        seq = core_row0 * (core_w - 1) + [border_color] * (core_w - 1) + [bg]
    else:
        seq = core_row0 + [bg]
    P = len(seq)
    
    # Fill output
    result = np.full_like(g, bg)
    for r in range(rows):
        for c in range(cols):
            d = (c - r) % P
            result[r, c] = seq[d]
    return result.tolist()
