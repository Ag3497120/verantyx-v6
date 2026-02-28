def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find key region: bounding box of cells that are non-zero and non-5
    key_cells = [(r, c) for r in range(rows) for c in range(cols) 
                 if g[r, c] != 0 and g[r, c] != 5]
    if not key_cells:
        return grid
    
    kr_min = min(r for r, c in key_cells)
    kr_max = max(r for r, c in key_cells)
    kc_min = min(c for r, c in key_cells)
    kc_max = max(c for r, c in key_cells)
    
    key_data = g[kr_min:kr_max+1, kc_min:kc_max+1]
    key_rows = kr_max - kr_min + 1
    key_cols = kc_max - kc_min + 1
    
    # Find 5-template grid: groups of consecutive 5-rows and 5-columns
    def group_indices(indices):
        if not indices:
            return []
        groups = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx > prev + 1:
                groups.append((start, prev))
                start = idx
            prev = idx
        groups.append((start, prev))
        return groups
    
    five_rows_all = sorted(set(r for r in range(rows) if any(g[r, c] == 5 for c in range(cols))))
    five_cols_all = sorted(set(c for c in range(cols) if any(g[r, c] == 5 for r in range(rows))))
    
    row_groups = group_indices(five_rows_all)
    col_groups = group_indices(five_cols_all)
    
    # Fill each template slot with corresponding key color
    for ri, (rstart, rend) in enumerate(row_groups):
        for ci, (cstart, cend) in enumerate(col_groups):
            kr = ri % key_rows
            kc = ci % key_cols
            color = key_data[kr, kc]
            if color != 0:
                for r in range(rstart, rend+1):
                    for c in range(cstart, cend+1):
                        if result[r, c] == 5:
                            result[r, c] = color
    
    return result.tolist()
