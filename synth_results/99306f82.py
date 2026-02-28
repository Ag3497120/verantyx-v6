def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from math import sqrt
    
    rect_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    rr = [r for r,c in rect_cells]
    rc = [c for r,c in rect_cells]
    r_min, r_max = min(rr), max(rr)
    c_min, c_max = min(rc), max(rc)
    
    markers = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) 
               if grid[r][c] != 0 and grid[r][c] != 1]
    # Sort by distance from rect - farthest first (outermost layer)
    markers_sorted = sorted(markers, key=lambda m: sqrt((m[0]-r_min)**2 + (m[1]-c_min)**2), reverse=True)
    
    result = [row[:] for row in grid]
    
    inner_r_min = r_min + 1
    inner_r_max = r_max - 1
    inner_c_min = c_min + 1
    inner_c_max = c_max - 1
    
    for layer_idx, (mr, mc, mv) in enumerate(markers_sorted):
        lr_min = inner_r_min + layer_idx
        lr_max = inner_r_max - layer_idx
        lc_min = inner_c_min + layer_idx
        lc_max = inner_c_max - layer_idx
        if lr_min > lr_max or lc_min > lc_max:
            break
        for r in range(lr_min, lr_max + 1):
            for c in range(lc_min, lc_max + 1):
                result[r][c] = mv
    
    return result
