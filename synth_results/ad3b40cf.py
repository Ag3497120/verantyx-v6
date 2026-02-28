def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find separator line (single column or row of a specific value)
    # Look for vertical separator
    sep_val = None
    sep_col = None
    for c in range(cols):
        col_vals = set(grid[r][c] for r in range(rows))
        if len(col_vals) == 1:
            v = grid[0][c]
            if v != bg:
                sep_val = v
                sep_col = c
                break
    
    # Find colored non-bg, non-sep block
    block_val = None
    for row in grid:
        for v in row:
            if v not in (bg, sep_val) and v is not None:
                block_val = v
                break
        if block_val: break
    
    if sep_col is None or block_val is None:
        return grid
    
    block_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==block_val]
    
    result = [row[:] for row in grid]
    
    # Reflect block across sep_col
    for r, c in block_cells:
        dist = c - sep_col
        nc = sep_col - dist
        if 0 <= nc < cols:
            result[r][nc] = block_val
    
    return result
