import numpy as np
from collections import Counter

def transform(grid):
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find the most common non-zero, non-2 color (background)
    all_vals = [grid[i][j] for i in range(h) for j in range(w) if grid[i][j] not in [0, 2]]
    if not all_vals:
        return grid
    
    bg_color = Counter(all_vals).most_common(1)[0][0]
    
    # Find all positions with 2s
    twos_set = set((i, j) for i in range(h) for j in range(w) if grid[i][j] == 2)
    
    # For each bg_color cell, check if it's adjacent to any 2
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg_color:
                # Check if adjacent to a 2
                adjacent_to_two = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in twos_set:
                        adjacent_to_two = True
                        break
                
                if adjacent_to_two:
                    grid[i][j] = 4
    
    return grid
