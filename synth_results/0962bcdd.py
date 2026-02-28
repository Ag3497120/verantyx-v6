def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    # Find all cross centers: a cell (r,c) with value V where all 4 neighbors have a different value A
    crosses = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            v = g[r, c]
            if v == 0:
                continue
            # Check all 4 neighbors
            neighbors = [g[r-1,c], g[r+1,c], g[r,c-1], g[r,c+1]]
            if all(n != 0 and n != v for n in neighbors) and len(set(neighbors)) == 1:
                arm_color = neighbors[0]
                crosses.append((r, c, v, arm_color))
    
    if not crosses:
        return grid
    
    for cr, cc, C, A in crosses:
        # Place star pattern
        result[cr, cc] = C
        
        # Arms N/S/E/W at distances 1 and 2: A
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            for d in [1, 2]:
                nr, nc = cr + dr*d, cc + dc*d
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr, nc] = A
        
        # Diagonals at distances 1 and 2: C
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            for d in [1, 2]:
                nr, nc = cr + dr*d, cc + dc*d
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr, nc] = C
    
    return result.tolist()
