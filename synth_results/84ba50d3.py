def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    H, W = g.shape
    
    bg = int(np.bincount(g.flatten()).argmax())
    
    sep_row = None; sep_val = None
    for r in range(H):
        row = g[r]
        u = np.unique(row)
        if len(u) == 1 and u[0] != bg:
            sep_row = r; sep_val = int(u[0]); break
    if sep_row is None: return grid
    
    below_size = H - sep_row - 1
    above_size = sep_row
    
    content_mask = (g != bg) & (g != sep_val)
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(content_mask, structure=struct)
    
    out = np.full_like(g, bg)
    out[sep_row] = sep_val
    
    for i in range(1, n+1):
        comp_mask = labeled == i
        rows_in = np.where(comp_mask.any(axis=1))[0]
        cols_in = np.where(comp_mask.any(axis=0))[0]
        width = int(cols_in.max() - cols_in.min() + 1)
        
        is_above = rows_in.min() < sep_row
        
        if is_above:
            if width == 1:
                # Single-column crossing component: goes to below side
                col = int(cols_in[0])
                content_rows = sorted([r for r in rows_in], key=lambda r: sep_row - r)
                occupied = {}
                for r in content_rows:
                    dist = sep_row - r
                    target = min(dist, below_size)
                    while target >= 1 and target in occupied:
                        target -= 1
                    if target >= 1:
                        occupied[target] = g[r, col]
                for dist, val in occupied.items():
                    out[sep_row + dist, col] = val
                out[sep_row, col] = bg  # hole in separator
            else:
                # Multi-column non-crossing: fill from sep-1 going DOWN
                for col in cols_in:
                    col_cells = sorted([r for r in range(sep_row) if comp_mask[r, col]],
                                       key=lambda r: sep_row - r)  # closest first
                    for j, r in enumerate(col_cells):
                        pos = sep_row - 1 - j + 2 * max(0, j - 0)  # simple: start at sep-1, go down
                        # Actually: pos = (sep_row - 1) + j (going down from sep-1)
                        pos = (sep_row - 1) + j
                        if 0 <= pos < H:
                            if pos == sep_row:
                                out[sep_row, col] = g[r, col]
                            else:
                                out[pos, col] = g[r, col]
        else:
            # Below separator
            if width == 1:
                col = int(cols_in[0])
                content_rows = sorted([r for r in rows_in], key=lambda r: r - sep_row)
                occupied = {}
                for r in content_rows:
                    dist = r - sep_row
                    target = min(dist, above_size)
                    while target >= 1 and target in occupied:
                        target -= 1
                    if target >= 1:
                        occupied[target] = g[r, col]
                for dist, val in occupied.items():
                    out[sep_row - dist, col] = val
                out[sep_row, col] = bg
            else:
                for col in cols_in:
                    col_cells = sorted([r for r in range(sep_row+1, H) if comp_mask[r, col]],
                                       key=lambda r: r - sep_row)  # closest first
                    for j, r in enumerate(col_cells):
                        pos = (sep_row + 1) - j  # going UP from sep+1
                        if 0 <= pos < H:
                            if pos == sep_row:
                                out[sep_row, col] = g[r, col]
                            else:
                                out[pos, col] = g[r, col]
    return out.tolist()
