def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from collections import Counter
    g = np.array(grid)
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    colors = set(v for v in g.flatten() if v != bg)
    
    # Use 8-connectivity to find groups
    struct = np.ones((3,3), dtype=int)
    
    color_groups = {}
    for color in colors:
        mask = (g == color).astype(int)
        labeled, num = label(mask, structure=struct)
        groups = []
        for i in range(1, num+1):
            cells = [tuple(x) for x in np.argwhere(labeled == i)]
            groups.append(cells)
        color_groups[color] = groups
    
    # The "repeated" shape is the one appearing the most times
    best_color = max(colors, key=lambda c: len(color_groups[c]))
    
    # Output bounding box of one instance
    cells = color_groups[best_color][0]
    rs = [r for r,c in cells]
    cs = [c for r,c in cells]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)
    return g[r_min:r_max+1, c_min:c_max+1].tolist()
