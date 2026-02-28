import numpy as np

def transform(grid):
    h, w = grid.shape
    output = np.zeros_like(grid)
    
    # Find connected components of same non-zero color (4-connectivity)
    visited = np.zeros((h, w), dtype=bool)
    rectangles = []  # each: (color, top, bottom, left, right)
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                # expand right
                right = c
                while right + 1 < w and grid[r, right + 1] == color:
                    right += 1
                # expand down
                bottom = r
                while bottom + 1 < h and all(grid[bottom + 1, col] == color for col in range(c, right + 1)):
                    bottom += 1
                # mark all in this rectangle as visited
                for rr in range(r, bottom + 1):
                    for cc in range(c, right + 1):
                        visited[rr, cc] = True
                rectangles.append((color, r, bottom, c, right))
    
    # Sort rectangles by top row (so topmost fall first? Actually in examples, topmost in input stays topmost after fall)
    # We'll process in original top-row order to preserve vertical order in each column group.
    rectangles.sort(key=lambda x: x[1])
    
    # For each rectangle, compute how far it can fall
    for color, top, bottom, left, right in rectangles:
        height = bottom - top + 1
        width = right - left + 1
        
        # Find drop distance
        max_drop = h - bottom - 1  # to bottom edge initially
        for drop in range(1, max_drop + 1):
            # Check if cells below are empty in all columns of rectangle
            can_drop = True
            for col in range(left, right + 1):
                if output[bottom + drop, col] != 0:
                    can_drop = False
                    break
            if not can_drop:
                max_drop = drop - 1
                break
        
        new_top = top + max_drop
        new_bottom = bottom + max_drop
        # Place in output
        for rr in range(new_top, new_bottom + 1):
            for cc in range(left, right + 1):
                output[rr, cc] = color
    
    return output