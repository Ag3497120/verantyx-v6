import numpy as np
from collections import Counter, deque

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    
    # Find shape color
    shape_color = None
    for v in g.flatten():
        if v != bg:
            shape_color = int(v)
            break
    if shape_color is None:
        return out.tolist()
    
    # Flood fill from edges to find outside bg cells
    outside = np.zeros((H, W), dtype=bool)
    queue = deque()
    for r in range(H):
        for c in [0, W-1]:
            if g[r, c] == bg and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(W):
        for r in [0, H-1]:
            if g[r, c] == bg and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not outside[nr, nc] and g[nr, nc] == bg:
                outside[nr, nc] = True
                queue.append((nr, nc))
    
    # Find connected components of shape_color (4-connected)
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or g[r, c] != shape_color:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cc < 0 or cr >= H or cc >= W:
                    continue
                if visited[cr, cc] or g[cr, cc] != shape_color:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((cr+dr, cc+dc))
            components.append(set(cells))
    
    # For each component, check if it has interior holes
    for comp in components:
        # Find all bg cells that are inside this component (not reachable from outside)
        has_holes = False
        for r, c in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == bg and not outside[nr, nc]:
                    has_holes = True
                    break
            if has_holes:
                break
        
        if has_holes:
            # Shape with holes: cells → 8, holes → 6
            for r, c in comp:
                out[r, c] = 8
            # Find interior holes for this component
            for r in range(H):
                for c in range(W):
                    if g[r, c] == bg and not outside[r, c]:
                        # Check if adjacent to this component
                        for r2, c2 in comp:
                            if abs(r-r2) + abs(c-c2) <= 2:  # rough check
                                out[r, c] = 6
                                break
        
        # Border: outside bg cells adjacent (8-connected) to component → 2
        for r, c in comp:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == bg and outside[nr, nc]:
                        out[nr, nc] = 2
    
    return out.tolist()
