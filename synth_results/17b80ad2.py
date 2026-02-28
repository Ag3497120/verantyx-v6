def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find the "sync" row: last non-zero row in the grid
    sync_row = -1
    for r in range(rows - 1, -1, -1):
        if any(g[r] != 0):
            sync_row = r
            break
    
    if sync_row == -1:
        return grid
    
    # Find the sync value (value that appears in the sync row)
    sync_vals = set(g[sync_row, c] for c in range(cols) if g[sync_row, c] != 0)
    
    # Find main columns (those with a sync value at the sync row)
    main_cols = set(c for c in range(cols) if g[sync_row, c] in sync_vals and g[sync_row, c] != 0)
    
    # For each main column, fill upward from each non-zero cell
    for c in main_cols:
        result[:, c] = 0  # clear column first
        nz = [(r, int(g[r, c])) for r in range(rows) if g[r, c] != 0]
        if not nz:
            continue
        prev_r = -1
        for r, v in nz:
            for fill_r in range(prev_r + 1, r + 1):
                result[fill_r, c] = v
            prev_r = r
        # rows after last non-zero stay 0
        for fill_r in range(prev_r + 1, rows):
            result[fill_r, c] = 0
    
    return result.tolist()
