def transform(grid):
    return _solve(grid)

def solve_7ddcd7ec(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find the solid block (NxN rectangle fully filled)
    non_zero = [(r,c,g[r][c]) for r in range(H) for c in range(W) if g[r][c] != 0]
    if not non_zero: return grid
    
    color = non_zero[0][2]
    positions = [(r,c) for r,c,v in non_zero]
    rows = [r for r,c in positions]
    cols = [c for r,c in positions]
    
    # Find the solid block (all cells in its bbox are filled)
    # Try to find a rectangular sub-region that's fully filled
    from scipy.ndimage import label
    arr = np.array([[1 if g[r][c]==color else 0 for c in range(W)] for r in range(H)])
    labeled, n = label(arr)
    
    # Find the block (largest connected component that's mostly a rectangle)
    block = None
    arms = []
    for lbl in range(1, n+1):
        pos = list(zip(*np.where(labeled==lbl)))
        rows_l = [p[0] for p in pos]
        cols_l = [p[1] for p in pos]
        r0, r1 = min(rows_l), max(rows_l)+1
        c0, c1 = min(cols_l), max(cols_l)+1
        area = len(pos)
        bbox_area = (r1-r0)*(c1-c0)
        if area == bbox_area and bbox_area > 1:
            block = (r0, r1, c0, c1)
        else:
            arms.extend(pos)
    
    if block is None: return grid
    
    br0, br1, bc0, bc1 = block
    
    # Extend each arm diagonally
    for (ar, ac) in arms:
        # Direction from nearest block corner to this arm cell
        # Find which corner of the block is closest
        # Actually: find direction from block corner through arm cell
        corners = [(br0-1, bc0-1), (br0-1, bc1), (br1, bc0-1), (br1, bc1)]
        # Find which corner the arm is extending from
        for cr, cc in corners:
            # Check if the arm is one diagonal step from this corner
            if abs(ar-cr) == 1 and abs(ac-cc) == 1:
                dr = ar - cr
                dc = ac - cc
                # Continue in this direction
                nr, nc = ar+dr, ac+dc
                while 0<=nr<H and 0<=nc<W:
                    out[nr][nc] = color
                    nr += dr; nc += dc
                break
    
    return out


_solve = solve_7ddcd7ec
