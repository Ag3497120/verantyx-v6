def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from collections import Counter
    g = np.array(grid)
    rows, cols = g.shape
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    values = set(v for v in g.flatten() if v != bg)
    
    if len(values) < 2:
        return grid
    
    # Find solid rectangle (fills bounding box entirely with one color)
    rect_color = None
    rect_bounds = None
    
    for color in values:
        mask = (g == color).astype(int)
        labeled, num = label(mask)
        cells = np.argwhere(labeled == 1)
        if len(cells) == 0:
            continue
        rs, cs = cells[:, 0], cells[:, 1]
        r_min, r_max, c_min, c_max = rs.min(), rs.max(), cs.min(), cs.max()
        bbox_area = (r_max - r_min + 1) * (c_max - c_min + 1)
        if len(cells) == bbox_area and len(cells) > 4:
            rect_color = color
            rect_bounds = (r_min, r_max, c_min, c_max)
            break
    
    if rect_color is None:
        return grid
    
    other_colors = values - {rect_color}
    if not other_colors:
        return grid
    other_color = other_colors.pop()
    
    visible = set((r, c) for r in range(rows) for c in range(cols) if g[r, c] == other_color)
    if not visible:
        return grid
    
    # Find symmetry axis: try all possible half-integer column positions
    # For each axis c_axis = k + 0.5, mirror c → k + k + 1 - c = 2k+1-c
    best_axis = None
    best_score = -1
    
    for k in range(cols - 1):
        # axis at k+0.5 → mirror_c = 2k+1 - c
        score = 0
        for r, c in visible:
            mc = 2*k+1 - c
            if (r, mc) in visible:
                score += 1
        if score > best_score:
            best_score = score
            best_axis = 2*k+1  # c_axis_num for half-integer: mirror_c = c_axis_num - c
    
    if best_axis is None:
        return [[bg]*cols]*rows
    
    result = [[bg]*cols for _ in range(rows)]
    for r, c in visible:
        result[r][c] = other_color
        mc = best_axis - c
        if 0 <= mc < cols:
            result[r][mc] = other_color
    
    return result
