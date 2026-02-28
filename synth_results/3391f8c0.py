def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    bg = 0
    values = set(v for r in grid for v in r if v != bg)
    
    struct8 = np.ones((3,3), dtype=int)
    
    def get_components(color, struct=None):
        mask = (g == color).astype(int)
        labeled, num = label(mask, structure=struct)
        comps = []
        for i in range(1, num+1):
            cells = sorted(map(tuple, np.argwhere(labeled == i)))
            comps.append(cells)
        return comps
    
    def bbox_center(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        return (min(rs)+max(rs)+1)//2, (min(cs)+max(cs)+1)//2
    
    # Find template color (single 8-connected component)
    template_color = None
    stamp_color = None
    template_cells = None
    stamp_groups = None
    
    for v in values:
        comps8 = get_components(v, struct=struct8)
        if len(comps8) == 1:
            template_color = v
            template_cells = comps8[0]
        else:
            stamp_color = v
            stamp_groups = comps8
    
    if template_color is None or stamp_color is None:
        return grid
    
    # Template center and relative positions
    t_cr, t_cc = bbox_center(template_cells)
    template_rel = [(r - t_cr, c - t_cc) for r, c in template_cells]
    
    # Stamp groups: for each group, place template at bbox center
    for group in stamp_groups:
        g_cr, g_cc = bbox_center(group)
        for dr, dc in template_rel:
            nr, nc = g_cr + dr, g_cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr, nc] = template_color
    
    # Place first stamp group centered at template center
    first_group = stamp_groups[0]
    f_cr, f_cc = bbox_center(first_group)
    for r, c in first_group:
        nr, nc = t_cr + (r - f_cr), t_cc + (c - f_cc)
        if 0 <= nr < rows and 0 <= nc < cols:
            result[nr, nc] = stamp_color
    
    return result.tolist()
