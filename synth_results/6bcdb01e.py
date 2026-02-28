def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [list(row) for row in grid]
    
    init_3 = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 3]
    if not init_3:
        return grid
    
    rows = [r for r, c in init_3]
    cols = [c for r, c in init_3]
    
    def turn_left(dr, dc): return dc, -dr
    def turn_right(dr, dc): return -dc, dr
    def in_bounds(r, c): return 0 <= r < H and 0 <= c < W
    def is_wall(r, c): return not in_bounds(r, c) or grid[r][c] == 8
    
    # Determine start and direction
    if max(rows) == min(rows):
        # Horizontal line - try right end first, then left
        candidates = [
            (rows[0], max(cols) + 1, 0, 1),   # from right end, going right
            (rows[0], min(cols) - 1, 0, -1),  # from left end, going left
        ]
    else:
        # Vertical line - try top end (going up), then bottom (going down)
        candidates = [
            (min(rows) - 1, cols[0], -1, 0),  # from top, going up
            (max(rows) + 1, cols[0], 1, 0),   # from bottom, going down
        ]
    
    # Pick the candidate that starts in bounds
    start_r, start_c, dr, dc = next(
        (sr, sc, d, dd) for sr, sc, d, dd in candidates if in_bounds(sr, sc)
    )
    r, c = start_r, start_c
    
    for _ in range(H * W * 4):
        if not in_bounds(r, c) or grid[r][c] == 8:
            break
        if grid[r][c] != 3:
            out[r][c] = 3
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            break
        if grid[nr][nc] == 8:
            ld, ldc = turn_left(dr, dc)
            rd, rdc = turn_right(dr, dc)
            if not is_wall(r + ld, c + ldc):
                dr, dc = ld, ldc
            elif not is_wall(r + rd, c + rdc):
                dr, dc = rd, rdc
            else:
                break
            r, c = r + dr, c + dc
        else:
            r, c = nr, nc
    
    return out
