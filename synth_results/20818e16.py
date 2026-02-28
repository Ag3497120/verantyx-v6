import numpy as np

def find_largest_rectangle(grid):
    h, w = grid.shape
    best = None
    best_area = 0
    
    for top in range(h):
        for bottom in range(top, h):
            for left in range(w):
                for right in range(left, w):
                    # Check if rectangle is uniform
                    subgrid = grid[top:bottom+1, left:right+1]
                    if np.all(subgrid == subgrid[0,0]):
                        # Check if touches at least two edges
                        touches = 0
                        if top == 0 or bottom == h-1:
                            touches += 1
                        if left == 0 or right == w-1:
                            touches += 1
                        if touches >= 2:
                            area = (bottom-top+1)*(right-left+1)
                            if area > best_area:
                                best_area = area
                                best = (top, bottom, left, right, subgrid[0,0])
    return best

def transform(grid):
    grid = np.array(grid)
    result = []
    
    while True:
        rect = find_largest_rectangle(grid)
        if rect is None:
            break
        top, bottom, left, right, color = rect
        result.append((bottom-top+1, right-left+1, color))
        # Remove rectangle by setting to a unique value (e.g., -1)
        grid[top:bottom+1, left:right+1] = -1
    
    # Sort by area descending
    result.sort(key=lambda x: x[0]*x[1], reverse=True)
    
    # Build output grid
    if not result:
        return []
    
    # Determine output size: sum of heights and widths minus overlaps
    total_h = sum(h for h, w, _ in result) - (len(result)-1)
    total_w = sum(w for h, w, _ in result) - (len(result)-1)
    
    out_grid = np.zeros((total_h, total_w), dtype=int)
    
    # Place rectangles diagonally
    r_offset = 0
    c_offset = 0
    for h, w, color in result:
        out_grid[r_offset:r_offset+h, c_offset:c_offset+w] = color
        r_offset += h - 1
        c_offset += w - 1
    
    return out_grid.tolist()