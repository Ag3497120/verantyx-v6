import numpy as np
from collections import defaultdict

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    # Find background (most common value, which is 0)
    bg = 0
    # Find non-background value
    nz = [(r,c) for r in range(h) for c in range(w) if g[r,c] != bg]
    
    if not nz:
        return grid
    
    # Find connected diagonal paths
    visited = set()
    
    def find_path(r, c, val):
        path = []
        # Follow the diagonal path in all 4 diagonal directions
        # Collect all connected diagonal cells
        stack = [(r,c)]
        seen = {(r,c)}
        while stack:
            cr, cc = stack.pop()
            path.append((cr, cc))
            for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = cr+dr, cc+dc
                if 0<=nr<h and 0<=nc<w and g[nr,nc]==val and (nr,nc) not in seen:
                    seen.add((nr,nc))
                    stack.append((nr,nc))
        return path
    
    # Find the non-background value
    val = g[nz[0][0], nz[0][1]]
    
    visited_cells = set()
    for r, c in nz:
        if (r,c) not in visited_cells:
            path = find_path(r, c, val)
            visited_cells.update(path)
            # Sort path by row, then col
            path.sort()
            # Alternate: keep first as val, change every other to 4
            for i, (pr, pc) in enumerate(path):
                if i % 2 == 1:
                    out[pr, pc] = 4
    
    return out.tolist()
