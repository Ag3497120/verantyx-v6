def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find cells with values 2, 3, 4
    pos = {}
    for r in range(rows):
        for c in range(cols):
            v = g[r, c]
            if v in (2, 3, 4):
                pos[v] = (r, c)
    
    if 4 not in pos:
        return grid
    
    r4, c4 = pos[4]
    
    # Path from 4 to cell with value 3 (horizontal-first, elbow at (r4, c3))
    if 3 in pos:
        r3, c3 = pos[3]
        # Horizontal leg: from (r4, c4) to (r4, c3)
        min_c, max_c = min(c4, c3), max(c4, c3)
        for c in range(min_c, max_c + 1):
            if result[r4, c] == 0:
                result[r4, c] = 5
        # Vertical leg: from (r4, c3) to (r3, c3)
        min_r, max_r = min(r4, r3), max(r4, r3)
        for r in range(min_r, max_r + 1):
            if result[r, c3] == 0:
                result[r, c3] = 5
    
    # Path from 4 to cell with value 2 (vertical-first, elbow at (r2, c4))
    if 2 in pos:
        r2, c2 = pos[2]
        # Vertical leg: from (r4, c4) to (r2, c4)
        min_r, max_r = min(r4, r2), max(r4, r2)
        for r in range(min_r, max_r + 1):
            if result[r, c4] == 0:
                result[r, c4] = 5
        # Horizontal leg: from (r2, c4) to (r2, c2)
        min_c, max_c = min(c4, c2), max(c4, c2)
        for c in range(min_c, max_c + 1):
            if result[r2, c] == 0:
                result[r2, c] = 5
    
    return result.tolist()
