def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find connected components of 8-cells
    visited = np.zeros((rows, cols), dtype=bool)
    
    def bfs(sr, sc):
        comp = []
        q = deque([(sr, sc)])
        visited[sr, sc] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and g[nr,nc]==8 and not visited[nr,nc]:
                    visited[nr,nc] = True
                    q.append((nr,nc))
        return comp
    
    components = []
    for r in range(rows):
        for c in range(cols):
            if g[r,c] == 8 and not visited[r,c]:
                components.append(bfs(r, c))
    
    for comp in components:
        comp_rows = [r for r,c in comp]
        comp_cols = [c for r,c in comp]
        min_r, max_r = min(comp_rows), max(comp_rows)
        min_c, max_c = min(comp_cols), max(comp_cols)
        
        center_r = (min_r + max_r) / 2.0
        center_c = (min_c + max_c) / 2.0
        
        # Find 2-cells within this frame's bounding box
        two_cells = [(r, c) for r in range(min_r, max_r+1) for c in range(min_c, max_c+1)
                     if g[r, c] == 2]
        
        # Add reflections
        for tr, tc in two_cells:
            new_positions = [
                (round(2*center_r - tr), tc),           # vertical reflection
                (tr, round(2*center_c - tc)),           # horizontal reflection
                (round(2*center_r - tr), round(2*center_c - tc)),  # 180° rotation
            ]
            for nr, nc in new_positions:
                if 0 <= nr < rows and 0 <= nc < cols and result[nr, nc] == 0:
                    result[nr, nc] = 2
    
    return result.tolist()
