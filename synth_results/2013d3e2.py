import numpy as np
from collections import Counter

def transform(grid):
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find the most common non-zero color (background)
    all_vals = [grid[i][j] for i in range(h) for j in range(w) if grid[i][j] not in [0, 2]]
    if not all_vals:
        return grid
    
    bg_color = Counter(all_vals).most_common(1)[0][0]
    
    # Find all positions with 2s
    twos = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == 2]
    if not twos:
        return grid
    
    # For each group of 2s, find bounding box
    visited = set()
    
    def get_connected_twos(sr, sc):
        stack = [(sr, sc)]
        component = []
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= h or c < 0 or c >= w:
                continue
            if grid[r][c] != 2:
                continue
            visited.add((r, c))
            component.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))
        return component
    
    for sr, sc in twos:
        if (sr, sc) in visited:
            continue
        
        component = get_connected_twos(sr, sc)
        if not component:
            continue
        
        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        
        # Replace background color with 4 in bounding box
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r][c] == bg_color:
                    grid[r][c] = 4
    
    return grid
