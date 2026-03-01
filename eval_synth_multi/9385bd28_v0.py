def transform(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Find 2x2 legend block (contains 3+ unique non-zero colors in a 2x2 area)
    legend = None
    legend_pos = None
    for r in range(rows - 1):
        for c in range(cols - 1):
            block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            unique_nonzero = set(v for v in block if v != 0)
            if len(unique_nonzero) >= 3:
                legend = block
                legend_pos = (r, c)
                break
        if legend: break
    
    if legend is None:
        # Try 2-unique legend
        for r in range(rows - 1):
            for c in range(cols - 1):
                block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
                if all(v != 0 for v in block):
                    unique = set(block)
                    if len(unique) >= 2:
                        legend = block
                        legend_pos = (r, c)
                        break
            if legend: break
    
    lr, lc = legend_pos
    inner_frame = legend[0]  # grid[lr][lc]
    inner_fill = legend[1]   # grid[lr][lc+1]
    outer_frame = legend[2]  # grid[lr+1][lc]
    outer_fill = legend[3]   # grid[lr+1][lc+1]
    
    # Legend cells should not be part of the main structure
    legend_cells = {(lr, lc), (lr, lc+1), (lr+1, lc), (lr+1, lc+1)}
    
    # Find bounding boxes
    outer_cells = [(r, c) for r in range(rows) for c in range(cols)
                   if grid[r][c] == outer_frame and (r, c) not in legend_cells]
    inner_cells = [(r, c) for r in range(rows) for c in range(cols)
                   if grid[r][c] == inner_frame and (r, c) not in legend_cells]
    
    if not outer_cells or not inner_cells:
        return out
    
    o_r1 = min(r for r, c in outer_cells)
    o_r2 = max(r for r, c in outer_cells)
    o_c1 = min(c for r, c in outer_cells)
    o_c2 = max(c for r, c in outer_cells)
    
    i_r1 = min(r for r, c in inner_cells)
    i_r2 = max(r for r, c in inner_cells)
    i_c1 = min(c for r, c in inner_cells)
    i_c2 = max(c for r, c in inner_cells)
    
    # Fill the outer rectangle
    for r in range(o_r1, o_r2 + 1):
        for c in range(o_c1, o_c2 + 1):
            if grid[r][c] == outer_frame or grid[r][c] == inner_frame:
                continue  # keep frame cells
            if i_r1 <= r <= i_r2 and i_c1 <= c <= i_c2:
                out[r][c] = inner_fill
            else:
                out[r][c] = outer_fill
    
    return out
