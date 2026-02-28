def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find key cells (non-0, non-8) and their positions/values
    key_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = g[r, c]
            if v != 0 and v != 8:
                key_cells[(r, c)] = v
    
    if not key_cells:
        return grid
    
    # Find bounding box of key cells
    key_rows = [r for r,c in key_cells]
    key_cols = [c for r,c in key_cells]
    kr0, kr1 = min(key_rows), max(key_rows)
    kc0, kc1 = min(key_cols), max(key_cols)
    key_h = kr1 - kr0 + 1
    key_w = kc1 - kc0 + 1
    
    # Build key grid
    key_grid = np.zeros((key_h, key_w), dtype=int)
    for (r, c), v in key_cells.items():
        key_grid[r - kr0, c - kc0] = v
    
    # Rotate 90° clockwise: (i,j) -> (j, key_h-1-i) in new shape (key_w, key_h)
    rotated = np.rot90(key_grid, k=-1)  # k=-1 = 90° CW
    rot_h, rot_w = rotated.shape
    
    # Find 8-cells bounding box
    eight_cells = [(r, c) for r in range(rows) for c in range(cols) if g[r, c] == 8]
    if not eight_cells:
        return grid
    
    er0 = min(r for r,c in eight_cells)
    ec0 = min(c for r,c in eight_cells)
    er1 = max(r for r,c in eight_cells)
    ec1 = max(c for r,c in eight_cells)
    eight_h = er1 - er0 + 1
    eight_w = ec1 - ec0 + 1
    
    # Block size
    block_h = eight_h // rot_h
    block_w = eight_w // rot_w
    
    if block_h == 0 or block_w == 0:
        return grid
    
    # Color each 8-cell based on rotated key meta-position
    for r, c in eight_cells:
        meta_r = (r - er0) // block_h
        meta_c = (c - ec0) // block_w
        if 0 <= meta_r < rot_h and 0 <= meta_c < rot_w:
            color = rotated[meta_r, meta_c]
            if color != 0:
                result[r, c] = color
    
    return result.tolist()
