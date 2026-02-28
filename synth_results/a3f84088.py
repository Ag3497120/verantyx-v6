import numpy as np
from collections import deque

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    # Find 0-cells enclosed by 5s (not reachable from border through 0-cells)
    reachable = set()
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if g[r,c] == 0 and (r,c) not in reachable:
                reachable.add((r,c)); q.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r,c] == 0 and (r,c) not in reachable:
                reachable.add((r,c)); q.append((r,c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and g[nr,nc]==0 and (nr,nc) not in reachable:
                reachable.add((nr,nc)); q.append((nr,nc))
    
    # Process each enclosed region
    enclosed_visited = set()
    
    for sr in range(h):
        for sc in range(w):
            if g[sr,sc] == 0 and (sr,sc) not in reachable and (sr,sc) not in enclosed_visited:
                # Flood fill this enclosed region
                region = []
                q2 = deque([(sr, sc)])
                enclosed_visited.add((sr, sc))
                while q2:
                    r, c = q2.popleft()
                    region.append((r, c))
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr,nc]==0 and (nr,nc) not in reachable and (nr,nc) not in enclosed_visited:
                            enclosed_visited.add((nr,nc))
                            q2.append((nr,nc))
                
                if not region:
                    continue
                
                # Find interior bounds
                rows = [r for r,c in region]
                cols = [c for r,c in region]
                R1, R2 = min(rows), max(rows)
                C1, C2 = min(cols), max(cols)
                
                # Compute d for each cell
                # Pattern: d%4: 0->2, 1->5, 2->0, 3->5
                # Special case: if max_d <= 3, d=3 maps to 0 (center is unreachable by second cycle)
                d_vals = {(r,c): min(r-R1, R2-r, c-C1, C2-c) for r,c in region}
                max_d = max(d_vals.values())
                
                color_map = {0: 2, 1: 5, 2: 0, 3: 5}
                
                for r, c in region:
                    d = d_vals[(r,c)]
                    color = color_map[d % 4]
                    # Special case: if max_d < 7 and d%4==3, the center isn't fully formed
                    if color == 5 and d % 4 == 3 and max_d < 7:
                        # Check if this is the "premature" end of cycle
                        # Count cells at this d
                        cells_at_d = sum(1 for dd in d_vals.values() if dd == d)
                        if cells_at_d < 4:  # Can't form a ring
                            color = 0
                    out[r, c] = color
    
    return out.tolist()
