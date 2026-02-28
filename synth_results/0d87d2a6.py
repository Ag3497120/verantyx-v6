def transform(grid):
    import numpy as np
    from collections import deque
    
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find all 1-cells
    one_cells = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 1]
    
    # Group by row and column
    by_row = {}
    by_col = {}
    for r, c in one_cells:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    
    # Lines to draw: for each row with 2+ 1-cells, draw horizontal line
    # For each col with 2+ 1-cells, draw vertical line
    lines = set()  # set of (r, c) cells on lines
    
    for r, cs in by_row.items():
        if len(cs) >= 2:
            min_c, max_c = min(cs), max(cs)
            for c in range(min_c, max_c+1):
                lines.add((r, c))
    
    for c, rs in by_col.items():
        if len(rs) >= 2:
            min_r, max_r = min(rs), max(rs)
            for r in range(min_r, max_r+1):
                lines.add((r, c))
    
    # Find connected components of 2-cells
    visited = np.zeros((rows, cols), dtype=bool)
    
    def bfs_component(sr, sc):
        comp = []
        q = deque([(sr, sc)])
        visited[sr, sc] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and g[nr,nc]==2 and not visited[nr,nc]:
                    visited[nr,nc] = True
                    q.append((nr,nc))
        return comp
    
    components = []
    for r in range(rows):
        for c in range(cols):
            if g[r,c] == 2 and not visited[r,c]:
                comp = bfs_component(r, c)
                components.append(comp)
    
    # Check which components are touched by lines
    touched = set()
    for i, comp in enumerate(components):
        for r, c in comp:
            if (r, c) in lines:
                touched.add(i)
                break
    
    # Convert touched components to 1
    for i, comp in enumerate(components):
        if i in touched:
            for r, c in comp:
                result[r, c] = 1
    
    # Draw lines (including cells not in 2-components)
    for r, c in lines:
        result[r, c] = 1
    
    return result.tolist()
