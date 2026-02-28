def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 7  # background
    mid_r = rows // 2
    mid_c = cols // 2
    
    # Find all unique non-bg values
    all_vals = set(v for row in grid for v in row if v != bg)
    
    # Each quadrant: get its cells
    quadrants = [
        [(r, c) for r in range(mid_r) for c in range(mid_c)],
        [(r, c) for r in range(mid_r) for c in range(mid_c+1, cols)],
        [(r, c) for r in range(mid_r+1, rows) for c in range(mid_c)],
        [(r, c) for r in range(mid_r+1, rows) for c in range(mid_c+1, cols)],
    ]
    
    result = [row[:] for row in grid]
    
    for quad in quadrants:
        q_vals = {grid[r][c] for r,c in quad if grid[r][c] != bg}
        missing = all_vals - q_vals
        bg_cells = [(r,c) for r,c in quad if grid[r][c] == bg]
        if missing and bg_cells:
            fill_val = list(missing)[0]
            for r,c in bg_cells:
                result[r][c] = fill_val
    
    return result
