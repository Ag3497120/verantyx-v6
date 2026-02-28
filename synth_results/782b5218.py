def transform(grid):
    return _solve(grid)

def solve_782b5218(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Background color = most common, fill color = second most common non-2
    vals = Counter(g.flatten())
    # 2 is the barrier
    non_two = {v: c for v, c in vals.items() if v != 2}
    if not non_two: return grid
    
    fill_color = max(non_two, key=lambda v: non_two[v])
    
    # For each column, find the topmost 2
    for c in range(W):
        first_two_row = None
        for r in range(H):
            if g[r, c] == 2:
                first_two_row = r
                break
        if first_two_row is None: continue
        # Above the 2: set to 0
        for r in range(first_two_row):
            out[r, c] = 0
        # The 2 stays
        # Below the 2: set to fill_color
        for r in range(first_two_row+1, H):
            out[r, c] = fill_color
    
    return out.tolist()


_solve = solve_782b5218
