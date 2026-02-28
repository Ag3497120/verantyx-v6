def transform(grid):
    return _solve(grid)

def solve_7ec998c9(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # Find the special cell
    special = [(r,c) for r in range(H) for c in range(W) if g[r][c] != bg]
    if not special: return grid
    sr, sc = special[0]
    special_val = int(g[sr, sc])
    
    out = g.copy().tolist()
    
    # Draw vertical line (1s at col sc, all rows except sr)
    for r in range(H):
        if r != sr:
            out[r][sc] = 1
    
    # At top edge (row 0): extend horizontally
    if sc >= W // 2:
        # Turn right
        for c in range(sc, W):
            out[0][c] = 1
    else:
        # Turn left
        for c in range(0, sc+1):
            out[0][c] = 1
    
    # At bottom edge (row H-1): opposite direction
    if sc >= W // 2:
        # Turn left
        for c in range(0, sc+1):
            out[H-1][c] = 1
    else:
        # Turn right
        for c in range(sc, W):
            out[H-1][c] = 1
    
    return out


_solve = solve_7ec998c9
