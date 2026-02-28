def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    R, C = g.shape
    bg = int(np.bincount(g.flatten()).argmax())
    out = np.full_like(g, bg)
    
    # Find clusters
    col_fills = {}  # col -> color
    row_fills = {}  # row -> color
    
    for v in np.unique(g):
        if v == bg: continue
        labeled, num = label(g == v)
        for cid in range(1, num+1):
            pos = np.argwhere(labeled == cid)
            rows_span = pos[:,0].max() - pos[:,0].min() + 1
            cols_span = pos[:,1].max() - pos[:,1].min() + 1
            if rows_span > cols_span:
                # Vertical: fill columns
                for c in np.unique(pos[:,1]):
                    col_fills[int(c)] = int(v)
            else:
                # Horizontal or square: fill rows
                for r in np.unique(pos[:,0]):
                    row_fills[int(r)] = int(v)
    
    # Apply column fills first, then row fills override
    for c, color in col_fills.items():
        out[:, c] = color
    for r, color in row_fills.items():
        out[r, :] = color
    
    # But what about bg cells? The original bg stays where no fill applies
    # Actually we need the column fills but with bg for cells covered by row fills
    # Let me rebuild: start with bg, apply col fills, apply row fills
    # (Already done above)
    
    return out.tolist()
