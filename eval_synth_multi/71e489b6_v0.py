def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    out = g.copy()
    
    # Find connected components
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if (r,c) in visited:
                continue
            val = g[r][c]
            cells = []
            q = deque([(r,c)])
            visited.add((r,c))
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and g[nr][nc] == val:
                        visited.add((nr,nc))
                        q.append((nr,nc))
            components.append((val, cells))
    
    # Find small components (defects)
    # Threshold: find the median component size for each value
    for val, cells in components:
        if len(cells) <= 5:  # small component = defect
            if val == 0:
                # Small 0-component in 1-region: add 7 border
                for r, c in cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols:
                                if out[nr][nc] == 1:
                                    out[nr][nc] = 7
            elif val == 1:
                # Small 1-component in 0-region: remove
                for r, c in cells:
                    out[r][c] = 0
    
    return out.tolist()
