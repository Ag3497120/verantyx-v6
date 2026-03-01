def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find the 2x2 block of non-zero values
    r1, c1 = -1, -1
    for r in range(R-1):
        for c in range(C-1):
            if grid[r][c] != 0 and grid[r][c+1] != 0 and grid[r+1][c] != 0 and grid[r+1][c+1] != 0:
                r1, c1 = r, c
                break
        if r1 != -1:
            break
    
    if r1 == -1:
        return result
    
    r2, c2 = r1+1, c1+1
    h, w = 2, 2
    
    TL = grid[r1][c1]
    TR = grid[r1][c2]
    BL = grid[r2][c1]
    BR = grid[r2][c2]
    
    above = r1
    below = R - 1 - r2
    left = c1
    right = C - 1 - c2
    
    # Top-left corner region -> fill with BR
    rr = min(above, h)
    rc = min(left, w)
    for r in range(r1 - rr, r1):
        for c in range(c1 - rc, c1):
            result[r][c] = BR
    
    # Top-right corner region -> fill with BL
    rr = min(above, h)
    rc = min(right, w)
    for r in range(r1 - rr, r1):
        for c in range(c2 + 1, c2 + 1 + rc):
            result[r][c] = BL
    
    # Bottom-left corner region -> fill with TR
    rr = min(below, h)
    rc = min(left, w)
    for r in range(r2 + 1, r2 + 1 + rr):
        for c in range(c1 - rc, c1):
            result[r][c] = TR
    
    # Bottom-right corner region -> fill with TL
    rr = min(below, h)
    rc = min(right, w)
    for r in range(r2 + 1, r2 + 1 + rr):
        for c in range(c2 + 1, c2 + 1 + rc):
            result[r][c] = TL
    
    return result
