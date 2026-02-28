def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    # Find border rows/cols of 1s
    # Row 0 and last: corners
    # The grid has a 1-border, corners have non-1 values, interior has 8s and 0s
    # Corners at (0,0), (0,-1), (-1,0), (-1,-1) might have colors
    # Actually: find rows that are all-1s (border)
    border_rows = [i for i in range(rows) if all(g[i,j]==1 for j in range(cols))]
    border_cols = [j for j in range(cols) if all(g[i,j]==1 for i in range(rows))]
    
    if not border_rows or not border_cols:
        # Simple: 1-pixel border
        r1, r2 = 0, rows-1
        c1, c2 = 0, cols-1
    else:
        r1, r2 = border_rows[0], border_rows[-1]
        c1, c2 = border_cols[0], border_cols[-1]
    
    # Extract corners (non-1 values at row border between col borders)
    # TL corner: at (r1, c1) area
    # Find corners: values that are not 0, not 1, not 8
    tl = tr = bl = br = 0
    for r in range(rows):
        for c in range(cols):
            v = g[r,c]
            if v != 0 and v != 1 and v != 8:
                # Determine which corner
                if r <= rows//2 and c <= cols//2:
                    tl = v
                elif r <= rows//2:
                    tr = v
                elif c <= cols//2:
                    bl = v
                else:
                    br = v
    
    # Extract interior (between borders)
    interior = g[r1+1:r2, c1+1:c2] if border_rows and border_cols else g[1:-1, 1:-1]
    ih, iw = interior.shape
    
    result = interior.copy()
    for r in range(ih):
        for c in range(iw):
            if interior[r,c] == 8:
                # Determine quadrant
                if r < ih//2 and c < iw//2:
                    result[r,c] = tl
                elif r < ih//2:
                    result[r,c] = tr
                elif c < iw//2:
                    result[r,c] = bl
                else:
                    result[r,c] = br
    return result.tolist()