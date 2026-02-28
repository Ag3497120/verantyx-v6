def transform(grid):
    return _solve(grid)

def solve_7b6016b9(grid):
    g = np.array(grid)
    H, W = g.shape
    out = np.full_like(g, 3)  # background -> 3
    
    # Copy non-zero cells (the 1s/8s framework)
    line_color = None
    for v in set(g.flatten()) - {0}:
        line_color = v; break
    
    for r in range(H):
        for c in range(W):
            if g[r,c] != 0:
                out[r,c] = g[r,c]
    
    # Find enclosed rectangular regions (bounded by line_color on all 4 sides)
    # A region is a connected component of 0s that is fully enclosed
    # BFS from border
    border_zeros = set()
    q = deque()
    for r in range(H):
        for c in [0, W-1]:
            if g[r,c] == 0:
                q.append((r,c)); border_zeros.add((r,c))
    for c in range(W):
        for r in [0, H-1]:
            if g[r,c] == 0 and (r,c) not in border_zeros:
                q.append((r,c)); border_zeros.add((r,c))
    
    while q:
        r,c = q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and g[nr,nc]==0 and (nr,nc) not in border_zeros:
                border_zeros.add((nr,nc))
                q.append((nr,nc))
    
    # Interior zeros = enclosed regions -> fill with 2
    for r in range(H):
        for c in range(W):
            if g[r,c] == 0 and (r,c) not in border_zeros:
                out[r,c] = 2
    
    return out.tolist()


_solve = solve_7b6016b9
