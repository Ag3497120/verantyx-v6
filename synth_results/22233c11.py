def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)
    
    # Find all 3 cells
    threes = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == 3]
    
    for r, c in threes:
        # Check for 2x2 block of 3s
        if (r + 1 < h and c + 1 < w and 
            grid[r, c] == 3 and grid[r, c+1] == 3 and 
            grid[r+1, c] == 3 and grid[r+1, c+1] == 3):
            # Place 8s at opposite corners of the bounding box
            out[r, c] = 8
            out[r+1, c] = 8
            out[r, c+1] = 8
            out[r+1, c+1] = 8
        else:
            # Single 3: place 8s at opposite diagonal corners of 3x3 centered on the 3
            center_r, center_c = r, c
            # Top-left and bottom-right
            tl_r, tl_c = center_r - 1, center_c - 1
            br_r, br_c = center_r + 1, center_c + 1
            if 0 <= tl_r < h and 0 <= tl_c < w:
                out[tl_r, tl_c] = 8
            if 0 <= br_r < h and 0 <= br_c < w:
                out[br_r, br_c] = 8
    
    # Keep original 3s
    out[grid == 3] = 3
    return out.tolist()