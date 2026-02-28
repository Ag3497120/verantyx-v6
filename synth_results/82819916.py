def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    # Find key row: full row (no zeros) with multiple unique values
    key_row = None
    for r in range(H):
        row = g[r]
        if 0 not in row and len(set(row.tolist())) > 1:
            key_row = row.copy()
            break
    if key_row is None:
        return grid
    key_vals = np.unique(key_row)
    
    # For each row with partial non-zero values, build a color mapping and fill
    for r in range(H):
        row = g[r]
        nz_pos = np.where(row != 0)[0]
        z_pos = np.where(row == 0)[0]
        if len(nz_pos) == 0 or len(z_pos) == 0:
            continue  # all zero or no zeros → skip
        # Build mapping from key values to partial row values
        color_map = {}
        valid = True
        for pos in nz_pos:
            kv = key_row[pos]
            rv = row[pos]
            if kv in color_map:
                if color_map[kv] != rv:
                    valid = False; break
            else:
                color_map[kv] = rv
        if not valid or len(color_map) < len(key_vals):
            continue
        # Fill zero positions using mapping
        new_row = row.copy()
        for pos in z_pos:
            kv = key_row[pos]
            if kv in color_map:
                new_row[pos] = color_map[kv]
        out[r] = new_row
    return out.tolist()
