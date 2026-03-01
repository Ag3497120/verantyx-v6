import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find background (most common color)
    counts = Counter(g.flatten())
    bg = counts.most_common(1)[0][0]
    
    # Find non-bg colors
    non_bg_colors = {}
    for r in range(H):
        for c in range(W):
            if g[r,c] != bg:
                v = int(g[r,c])
                if v not in non_bg_colors:
                    non_bg_colors[v] = []
                non_bg_colors[v].append((r,c))
    
    # Fill color = the color with fewest cells that's not the main boundary
    # Actually: find the single cell (or small cluster) that acts as seed
    # Sort by count; the one with fewest is likely the fill seed
    sorted_colors = sorted(non_bg_colors.items(), key=lambda x: len(x[1]))
    
    # The fill color is the one with fewest cells
    fill_color = sorted_colors[0][0]
    seeds = sorted_colors[0][1]
    
    # Flood fill from seeds, replacing bg with fill_color
    # Walls = all non-bg, non-fill cells
    visited = set()
    queue = list(seeds)
    for s in queue:
        visited.add(s)
    
    while queue:
        r, c = queue.pop(0)
        out[r, c] = fill_color
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if g[nr, nc] == bg:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return out.tolist()
