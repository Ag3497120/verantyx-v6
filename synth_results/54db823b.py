
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    
    # Find connected regions of 3s and 9s
    visited = set()
    regions = []
    
    def bfs(sr, sc):
        q = [(sr, sc)]
        cells = []
        while q:
            r, c = q.pop(0)
            if (r,c) in visited or r<0 or r>=h or c<0 or c>=w: continue
            if g[r,c] not in (3, 9): continue
            visited.add((r,c))
            cells.append((r,c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                q.append((r+dr, c+dc))
        return cells
    
    for r in range(h):
        for c in range(w):
            if g[r,c] in (3, 9) and (r,c) not in visited:
                cells = bfs(r, c)
                if cells:
                    n9 = sum(1 for r2,c2 in cells if g[r2,c2]==9)
                    regions.append((cells, n9))
    
    # Remove the region with fewest 9s
    if not regions:
        return g.tolist()
    
    min_n9 = min(n9 for _, n9 in regions)
    out = g.copy()
    for cells, n9 in regions:
        if n9 == min_n9:
            for r, c in cells:
                out[r, c] = 0
            break  # only remove one
    
    return out.tolist()
