def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    bg = 0
    values = set(v for r in grid for v in r if v != bg)
    
    # Find the "stamp" color (appears in multiple identical components)
    # Find the "template" (other colors forming a single pattern)
    
    def get_components(color):
        mask = (g == color).astype(int)
        labeled, num = label(mask)
        comps = []
        for i in range(1, num+1):
            cells = sorted(map(tuple, np.argwhere(labeled == i)))
            comps.append(cells)
        return comps
    
    def shape_of(cells):
        cells = sorted(cells)
        r0, c0 = cells[0]
        return frozenset((r-r0, c-c0) for r, c in cells)
    
    stamp_color = None
    stamp_comps = None
    
    for v in values:
        comps = get_components(v)
        if len(comps) >= 2:
            shapes = [shape_of(c) for c in comps]
            if len(set(shapes)) == 1:
                # All components have same shape - this is the stamp color
                stamp_color = v
                stamp_comps = comps
                break
    
    if stamp_color is None:
        return grid
    
    # Template = all non-bg, non-stamp cells
    template_cells = {}  # color -> list of (r,c)
    for v in values:
        if v != stamp_color:
            cells = list(map(tuple, np.argwhere(g == v)))
            if cells:
                template_cells[v] = cells
    
    if not template_cells:
        return grid
    
    # Find template bounding box top-left
    all_template = [(r, c) for cells in template_cells.values() for r, c in cells]
    t_r0 = min(r for r, c in all_template)
    t_c0 = min(c for r, c in all_template)
    
    # Template relative positions per color
    template_rel = {}
    for v, cells in template_cells.items():
        template_rel[v] = [(r - t_r0, c - t_c0) for r, c in cells]
    
    # For each stamp component, place template at its top-left
    for stamp_cells in stamp_comps:
        s_r0 = min(r for r, c in stamp_cells)
        s_c0 = min(c for r, c in stamp_cells)
        for v, rel_cells in template_rel.items():
            for dr, dc in rel_cells:
                nr, nc = s_r0 + dr, s_c0 + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr, nc] = v
    
    return result.tolist()
