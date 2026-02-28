def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    # Find divider column (column of 5s)
    div_col = None
    for c in range(cols):
        if all(g[r, c] == 5 for r in range(rows)):
            div_col = c
            break
    
    if div_col is None:
        return grid
    
    # Template is on the left of div_col
    template = g[:, :div_col].copy()
    # Right field has 1s as markers
    right = g[:, div_col+1:].copy()
    
    # Copy template to left
    result[:, :div_col] = template
    result[:, div_col] = 5
    
    t_rows, t_cols = template.shape
    
    # For each 1 in the right field, stamp the template centered on it
    # Find center of template (where it would be "anchored")
    # The template's "center" is roughly the center of mass of non-zero cells
    nz_cells = np.argwhere(template != 0)
    if len(nz_cells) > 0:
        cr, cc = nz_cells.mean(axis=0)
        cr, cc = int(round(cr)), int(round(cc))
    else:
        cr, cc = t_rows // 2, t_cols // 2
    
    # Find all 1s in right field
    marker_positions = np.argwhere(right == 1)
    
    roff = 0
    coff = div_col + 1
    
    for mr, mc in marker_positions:
        # Place template so its (cr,cc) cell aligns with marker
        dr = mr - cr
        dc = mc - cc
        for r in range(t_rows):
            for c in range(t_cols):
                nr = r + dr
                nc = c + dc + coff
                if 0 <= nr < rows and 0 <= nc < cols:
                    if template[r, c] != 0:
                        result[nr, nc] = template[r, c]
    
    return result.tolist()
