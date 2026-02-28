def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find the "3" groups and non-3/non-0 patterns
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    # Find all 3-groups
    threes = (g == 3)
    labeled3, n3 = label(threes, structure=struct)
    
    # Find all colored patterns (non-0, non-3)
    # Get unique colors
    unique_colors = [v for v in np.unique(g) if v not in [0, 3]]
    
    # Extract shapes for each color
    def get_shape(mask):
        positions = np.argwhere(mask)
        if len(positions) == 0: return frozenset()
        min_r, min_c = positions[:,0].min(), positions[:,1].min()
        return frozenset((int(r-min_r), int(c-min_c)) for r,c in positions)
    
    def get_all_rotations(shape):
        """Get all 4 rotations and 4 reflections of the shape"""
        variants = set()
        positions = list(shape)
        for flip in [False, True]:
            pos = [(r, -c) if flip else (r, c) for r, c in positions]
            for rot in range(4):
                if rot == 1: pos = [(-c, r) for r, c in pos]
                elif rot == 2: pos = [(-r, -c) for r, c in pos]
                elif rot == 3: pos = [(c, -r) for r, c in pos]
                min_r = min(r for r, c in pos)
                min_c = min(c for r, c in pos)
                normalized = frozenset((r-min_r, c-min_c) for r, c in pos)
                variants.add(normalized)
        return variants
    
    # Build color→shape mapping
    color_shapes = {}
    for color in unique_colors:
        mask = (g == color)
        # Split into connected components for this color
        lc, nc = label(mask, structure=struct)
        for i in range(1, nc+1):
            s = get_shape(lc == i)
            if s not in color_shapes:
                color_shapes[s] = color
    
    # For each 3-group, find matching colored shape
    for i in range(1, n3+1):
        mask3 = (labeled3 == i)
        shape3 = get_shape(mask3)
        rotations3 = get_all_rotations(shape3)
        
        # Find matching color
        matched_color = None
        for s, col in color_shapes.items():
            if s in rotations3:
                matched_color = col
                break
        
        if matched_color is not None:
            out[mask3] = matched_color
    
    return out.tolist()
