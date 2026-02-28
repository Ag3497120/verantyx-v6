def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    # Find rectangle color and cross color
    # Rectangle is a closed border shape
    # Cross is two perpendicular lines passing through the rectangle
    unique_vals = [v for v in np.unique(g) if v != 0]
    
    rect_color = None
    rect_bounds = None
    cross_color = None
    cross_row = None
    cross_col = None
    
    for v in unique_vals:
        # Check if v forms a rectangle border
        pos = np.argwhere(g == v)
        rmin, cmin = pos.min(0)
        rmax, cmax = pos.max(0)
        # Rect border: cells on the boundary of the bounding box
        rect_cells = set()
        for r in range(rmin, rmax+1):
            rect_cells.add((r, cmin))
            rect_cells.add((r, cmax))
        for c in range(cmin, cmax+1):
            rect_cells.add((rmin, c))
            rect_cells.add((rmax, c))
        pos_set = set(map(tuple, pos.tolist()))
        if pos_set == rect_cells:
            rect_color = v
            rect_bounds = (rmin, rmax, cmin, cmax)
    
    for v in unique_vals:
        if v == rect_color:
            continue
        pos = np.argwhere(g == v)
        rows = np.unique(pos[:, 0])
        cols = np.unique(pos[:, 1])
        # Cross: appears in all columns of one row and all rows of one col
        # Find which row and col the cross is at
        row_counts = np.bincount(pos[:, 0], minlength=R)
        col_counts = np.bincount(pos[:, 1], minlength=C)
        # The cross row is the full-width row, cross col is the full-height col
        max_row = np.argmax(row_counts)
        max_col = np.argmax(col_counts)
        if row_counts[max_row] > 1 and col_counts[max_col] > 1:
            cross_color = v
            cross_row = max_row
            cross_col = max_col
            break
    
    if rect_bounds is None or cross_color is None:
        return grid
    
    rmin, rmax, cmin, cmax = rect_bounds
    
    # Move cross to top edge (row=rmin) and right edge (col=cmax)
    out = np.zeros_like(g)
    
    # Draw rectangle
    for r in range(rmin, rmax+1):
        out[r, cmin] = rect_color
        out[r, cmax] = rect_color
    for c in range(cmin, cmax+1):
        out[rmin, c] = rect_color
        out[rmax, c] = rect_color
    
    # Draw cross at new position: horizontal at rmin, vertical at cmax
    for c in range(C):
        if out[rmin, c] == 0:
            out[rmin, c] = cross_color
    for r in range(R):
        if out[r, cmax] == 0:
            out[r, cmax] = cross_color
    
    return out.tolist()
