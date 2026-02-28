def transform(grid):
    return _solve(grid)

def solve_7f4411dc(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = 0
    out = np.zeros_like(g)
    
    # Find all connected components of non-zero cells
    from scipy.ndimage import label
    mask = (g != bg).astype(int)
    labeled, n = label(mask)
    
    for lbl in range(1, n+1):
        positions = list(zip(*np.where(labeled == lbl)))
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        area = len(positions)
        bbox_area = (r1-r0) * (c1-c0)
        
        # It's a solid rectangle if area == bbox_area and at least 2x2
        if area == bbox_area and (r1-r0) >= 2 and (c1-c0) >= 2:
            color = int(g[positions[0][0], positions[0][1]])
            out[r0:r1, c0:c1] = color
    
    return out.tolist()


_solve = solve_7f4411dc
