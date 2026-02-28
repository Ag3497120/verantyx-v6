def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [list(row) for row in grid]
    
    # Find frame (2s)
    twos = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 2]
    r0 = min(r for r, c in twos)
    r1 = max(r for r, c in twos)
    c0 = min(c for r, c in twos)
    c1 = max(c for r, c in twos)
    
    # Find scattered shape cells (non-2, non-0)
    shape_color = None
    shapes = [(r, c) for r in range(H) for c in range(W) if grid[r][c] not in (0, 2)]
    if not shapes:
        return grid
    shape_color = grid[shapes[0][0]][shapes[0][1]]
    
    # Remove original shapes
    for r, c in shapes:
        out[r][c] = 0
    
    # Reflect each shape cell into the frame
    for r, c in shapes:
        if c < c0:
            nc = 2 * c0 - c
            nr = r
        elif c > c1:
            nc = 2 * c1 - c
            nr = r
        elif r < r0:
            nr = 2 * r0 - r
            nc = c
        elif r > r1:
            nr = 2 * r1 - r
            nc = c
        else:
            nr, nc = r, c  # already inside
        
        if 0 <= nr < H and 0 <= nc < W:
            out[nr][nc] = shape_color
    
    return out
