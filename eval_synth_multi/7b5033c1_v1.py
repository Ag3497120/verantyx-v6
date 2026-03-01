def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    bg = g[0][0]  # background is most common / corner
    
    # Find all non-background cells
    non_bg = set()
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != bg:
                non_bg.add((r,c))
    
    if not non_bg:
        return grid
    
    # Build adjacency for the path
    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in non_bg:
                yield (nr, nc)
    
    # Find endpoints (cells with exactly 1 neighbor in the path)
    endpoints = []
    for cell in non_bg:
        n_count = sum(1 for _ in neighbors(*cell))
        if n_count == 1:
            endpoints.append(cell)
    
    if not endpoints:
        # Might be a loop, just pick any cell
        start = min(non_bg)
    else:
        start = min(endpoints)
    
    # Trace path from start
    path = [start]
    visited = {start}
    current = start
    while True:
        found = False
        for nb in neighbors(*current):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                current = nb
                found = True
                break
        if not found:
            break
    
    # Group consecutive same-colored cells
    result = []
    for r, c in path:
        color = g[r][c]
        result.append([int(color)])
    
    return result
