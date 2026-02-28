def transform(grid):
    import numpy as np
    from collections import deque
    
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # BFS from all boundary cells (treating 1s as walls) to find "outside" cells
    outside = np.zeros((rows, cols), dtype=bool)
    queue = deque()
    
    for r in range(rows):
        for c in [0, cols-1]:
            if g[r, c] != 1 and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows-1]:
            if g[r, c] != 1 and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not outside[nr, nc] and g[nr, nc] != 1:
                outside[nr, nc] = True
                queue.append((nr, nc))
    
    # "Interior" cells: NOT outside, and NOT a "boundary 1" (1-cell adjacent to outside)
    # Boundary 1: g[r,c]==1 and any neighbor is outside
    boundary1 = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if g[r, c] == 1:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and outside[nr, nc]:
                        boundary1[r, c] = True
                        break
    
    # Interior: everything that is not outside and not boundary1
    interior = ~outside & ~boundary1
    
    # Find interior connected components and their seed colors
    visited = np.zeros((rows, cols), dtype=bool)
    for sr in range(rows):
        for sc in range(cols):
            if interior[sr, sc] and not visited[sr, sc]:
                # BFS this component
                comp = []
                seed_color = 0
                q = deque([(sr, sc)])
                visited[sr, sc] = True
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    if g[r, c] not in (0, 1):
                        seed_color = g[r, c]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and interior[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                
                if seed_color != 0:
                    for r, c in comp:
                        result[r, c] = seed_color
    
    return result.tolist()
