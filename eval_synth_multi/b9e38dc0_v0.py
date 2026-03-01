import numpy as np
from collections import Counter, deque

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    
    # Find non-bg colors and their cell counts
    colors = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v != bg:
                if v not in colors:
                    colors[v] = []
                colors[v].append((r, c))
    
    if not colors:
        return out.tolist()
    
    # Fill color = the color with fewest cells (excluding the boundary)
    sorted_colors = sorted(colors.keys(), key=lambda c: len(colors[c]))
    fill_color = sorted_colors[0]
    fill_cells = colors[fill_color]
    
    # Flood fill from fill cells, treating all non-bg non-fill cells as walls
    visited = np.zeros((H, W), dtype=bool)
    queue = deque()
    
    for r, c in fill_cells:
        visited[r, c] = True
        queue.append((r, c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                v = int(g[nr, nc])
                if v == bg:
                    visited[nr, nc] = True
                    out[nr, nc] = fill_color
                    queue.append((nr, nc))
    
    return out.tolist()
