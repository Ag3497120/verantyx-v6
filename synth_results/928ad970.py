def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    # Find marker color (most common non-zero that appears 4 times scattered)
    # Actually: find the rect color (non-zero, forms a rect border) and marker color (5?)
    vals, cnts = np.unique(g, return_counts=True)
    nz = {v: c for v, c in zip(vals, cnts) if v != 0}
    
    # Marker color = 5 (or whichever appears 4 times spread out)
    # Rect color = the other non-zero color
    marker_color = None
    rect_color = None
    for v, c in nz.items():
        if c == 4:
            marker_color = v
        else:
            rect_color = v
    
    if marker_color is None or rect_color is None:
        return grid
    
    # Find marker positions
    markers = np.argwhere(g == marker_color)
    r_min, r_max = markers[:, 0].min(), markers[:, 0].max()
    c_min, c_max = markers[:, 1].min(), markers[:, 1].max()
    
    # Draw outer rectangle one step inside the bounding box
    out = g.copy()
    r1, r2 = r_min + 1, r_max - 1
    c1, c2 = c_min + 1, c_max - 1
    
    out[r1, c1:c2+1] = rect_color
    out[r2, c1:c2+1] = rect_color
    out[r1:r2+1, c1] = rect_color
    out[r1:r2+1, c2] = rect_color
    
    return out.tolist()
