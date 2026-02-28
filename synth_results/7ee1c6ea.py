def transform(grid):
    return _solve(grid)

def solve_7ee1c6ea(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    
    # Find the 5-border rectangle
    five_positions = [(r,c) for r in range(H) for c in range(W) if g[r][c]==5]
    if not five_positions: return grid
    
    rows = [p[0] for p in five_positions]
    cols = [p[1] for p in five_positions]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    
    # Find the two non-5, non-0 colors inside the border
    interior = [(r,c) for r in range(r0+1, r1) for c in range(c0+1, c1) if g[r][c] not in [0,5]]
    interior_colors = set(g[r][c] for r,c in interior)
    
    if len(interior_colors) < 2: return grid
    colors = list(interior_colors)
    a, b = colors[0], colors[1]
    
    # Swap a<->b inside the border
    out = [row[:] for row in g]
    for r in range(r0+1, r1):
        for c in range(c0+1, c1):
            if out[r][c] == a:
                out[r][c] = b
            elif out[r][c] == b:
                out[r][c] = a
    
    return out


_solve = solve_7ee1c6ea
