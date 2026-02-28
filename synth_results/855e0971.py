def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    # For each 0, find the contiguous same-color region around it
    # Then fill across the shorter dimension
    visited = np.zeros((H, W), bool)
    from collections import deque
    
    zeros = np.argwhere(g == 0)
    for zr, zc in zeros:
        if visited[zr, zc]:
            continue
        # BFS to find the background color block containing this 0
        # Actually: find the row band or col band by looking at context
        # Determine background color: look at neighbors
        bg = None
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = zr+dr, zc+dc
            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != 0:
                bg = g[nr, nc]
                break
        if bg is None:
            continue
        # Find the rectangular region of background color (or 0)
        # by expanding from this 0
        # Find row extent: rows where col zc has bg or 0
        r1 = zr
        while r1 > 0 and g[r1-1, zc] in [0, bg]: r1 -= 1
        r2 = zr
        while r2 < H-1 and g[r2+1, zc] in [0, bg]: r2 += 1
        # Find col extent: cols where row zr has bg or 0
        c1 = zc
        while c1 > 0 and g[zr, c1-1] in [0, bg]: c1 -= 1
        c2 = zc
        while c2 < W-1 and g[zr, c2+1] in [0, bg]: c2 += 1
        height = r2 - r1 + 1
        width = c2 - c1 + 1
        if height <= width:
            # Fill column within row band
            out[r1:r2+1, zc] = 0
        else:
            # Fill row within col band
            out[zr, c1:c2+1] = 0
        visited[zr, zc] = True
    return out.tolist()
