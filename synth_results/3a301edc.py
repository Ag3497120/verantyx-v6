def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find the bordered rectangle region
    # Background = 0; rectangle = border_color (non-0, appears in L-shape), interior = inner_color
    
    bg = 0
    nz = [(r, c) for r in range(rows) for c in range(cols) if g[r, c] != bg]
    if not nz:
        return grid
    
    rs = [r for r, c in nz]
    cs = [c for r, c in nz]
    r_min, r_max, c_min, c_max = min(rs), max(rs), min(cs), max(cs)
    
    # Inside the bounding box, find border color and interior color
    vals = Counter(g[r, c] for r, c in nz)
    # Border color = most common non-bg value
    border_color = vals.most_common(1)[0][0]
    # Interior color = the OTHER non-bg value (if any)
    interior_color = None
    for val, cnt in vals.items():
        if val != border_color:
            interior_color = val
            break
    
    if interior_color is None:
        return grid
    
    # Find interior dimensions
    int_cells = [(r, c) for r in range(rows) for c in range(cols) if g[r, c] == interior_color]
    int_rs = [r for r, c in int_cells]
    int_cs = [c for r, c in int_cells]
    int_h = max(int_rs) - min(int_rs) + 1
    int_w = max(int_cs) - min(int_cs) + 1
    
    # Fill exterior ring around the rectangle
    ext_r_min = max(0, r_min - int_h)
    ext_r_max = min(rows-1, r_max + int_h)
    ext_c_min = max(0, c_min - int_w)
    ext_c_max = min(cols-1, c_max + int_w)
    
    for r in range(ext_r_min, ext_r_max+1):
        for c in range(ext_c_min, ext_c_max+1):
            if not (r_min <= r <= r_max and c_min <= c <= c_max):
                result[r, c] = interior_color
    
    return result.tolist()
