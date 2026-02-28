def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find the "template" (large cluster of non-zero cells)
    # and isolated "marker" cells
    non_zero = [(r, c, g[r, c]) for r in range(rows) for c in range(cols) if g[r, c] != 0]
    
    if not non_zero:
        return grid
    
    # Find marker values that appear as single isolated cells
    # The template is the largest connected cluster
    from scipy.ndimage import label
    nz_mask = (g != 0).astype(int)
    labeled, num = label(nz_mask)
    
    # Find the largest component (template)
    comp_sizes = [(np.sum(labeled == i), i) for i in range(1, num+1)]
    comp_sizes.sort(reverse=True)
    
    template_comp = comp_sizes[0][1]
    template_cells = np.argwhere(labeled == template_comp)
    
    # Find the special value in the template (appears once, while other values appear more)
    # The special value is shared between template and marker
    template_values = {}
    for r, c in template_cells:
        v = g[r, c]
        template_values[v] = template_values.get(v, 0) + 1
    
    # Isolated marker cells (other components)
    marker_comps = [(sz, comp) for sz, comp in comp_sizes[1:] if sz == 1]
    
    for _, comp in marker_comps:
        mc = np.argwhere(labeled == comp)[0]
        mr, mc_col = mc[0], mc[1]
        marker_val = g[mr, mc_col]
        
        # Find where this value appears in the template
        template_matches = [(r, c) for r, c in template_cells if g[r, c] == marker_val]
        if not template_matches:
            continue
        tr, tc = template_matches[0]
        
        # Offset: place template so that (tr,tc) maps to (mr,mc_col)
        dr, dc = mr - tr, mc_col - tc
        
        # Copy template to new position (with marker cell becoming 0)
        for r, c in template_cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if r == tr and c == tc:
                    result[nr, nc] = 0  # marker position becomes 0
                else:
                    result[nr, nc] = g[r, c]
    
    return result.tolist()
