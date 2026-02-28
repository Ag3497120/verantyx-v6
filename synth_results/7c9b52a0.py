def transform(grid):
    return _solve(grid)

def solve_7c9b52a0(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # Find all rectangular "windows" (regions bounded by bg values)
    # Windows = connected regions of 0s within the bg
    non_bg = np.where(g != bg)
    if len(non_bg[0]) == 0: return grid
    
    # Find the "0-filled" rectangular windows
    # A window is a rectangle of 0s within the bg
    # Find connected components of non-bg cells
    from scipy.ndimage import label
    mask = (g != bg).astype(int)
    labeled, n = label(mask)
    
    windows = []
    for lbl in range(1, n+1):
        positions = list(zip(*np.where(labeled == lbl)))
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        windows.append((r0, r1, c0, c1, positions))
    
    # All windows should have the same size
    if not windows: return grid
    sizes = [(r1-r0, c1-c0) for r0,r1,c0,c1,_ in windows]
    target_size = Counter(sizes).most_common(1)[0][0]
    
    H_win, W_win = target_size
    result = [[0]*W_win for _ in range(H_win)]
    
    for r0, r1, c0, c1, positions in windows:
        if r1-r0 != H_win or c1-c0 != W_win: continue
        for r, c in positions:
            val = int(g[r, c])
            if val != bg and val != 0:
                result[r-r0][c-c0] = val
    
    return result


_solve = solve_7c9b52a0
