def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find 5-block boundaries
    five_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not five_cells:
        return grid
    min_r = min(r for r,c in five_cells)
    max_r = max(r for r,c in five_cells)
    min_c = min(c for r,c in five_cells)
    max_c = max(c for r,c in five_cells)
    
    # Find 2 and 8 positions
    pos2, pos8 = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2: pos2 = (r,c)
            if grid[r][c] == 8: pos8 = (r,c)
    
    if pos2 is None or pos8 is None:
        return grid
    
    r2, c2 = pos2
    r8, c8 = pos8
    
    # Clear originals
    result[r2][c2] = 7
    result[r8][c8] = 7
    
    # 2 moves to where 8 was
    result[r8][c8] = 2
    
    # 8 moves to outside corner of 5-block
    # Direction: if 2 is LEFT of 8, 8 goes to bottom-right exterior
    # if 2 is RIGHT of 8, 8 goes to bottom-left exterior
    if c2 < c8:
        # 2 is left of 8, 8 goes bottom-right
        result[max_r][max_c+1] = 8
    else:
        # 2 is right of 8, 8 goes bottom-left
        result[max_r][min_c-1] = 8
    
    return result
