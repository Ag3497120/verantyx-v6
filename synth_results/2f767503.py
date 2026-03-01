import numpy as np
from collections import deque

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    result = arr.copy()
    
    r5, c5 = np.where(arr == 5)
    r9, c9 = np.where(arr == 9)
    if len(r9) == 0 or len(r5) == 0: return grid
    
    nine_r, nine_c = r9[0], c9[0]
    five_cells = list(zip(r5.tolist(), c5.tolist()))
    
    # Background value (most common)
    bg = 7  # assuming 7 is background
    mark = 4  # the value to remove
    
    removed = set()
    
    # For each 5-cell, shoot ray from 9 THROUGH the 5-cell and beyond
    for (fr, fc) in five_cells:
        dr = fr - nine_r
        dc = fc - nine_c
        # Continue from 5-cell in same direction
        r, c = fr + dr, fc + dc
        while 0 <= r < rows and 0 <= c < cols:
            if arr[r, c] == mark:
                removed.add((r, c))
            r += dr
            c += dc
    
    # Cascade: flood fill from removed 4s to connected 4s
    queue = deque(removed)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in removed and arr[nr,nc] == mark:
                removed.add((nr,nc))
                queue.append((nr,nc))
    
    for (r, c) in removed:
        result[r, c] = bg
    
    return result.tolist()
