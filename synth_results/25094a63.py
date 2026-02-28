def transform(grid):
    import numpy as np
    
    g = np.array(grid)
    h, w = g.shape
    
    # For each color, find the SINGLE LARGEST uniform rectangle
    # Mark those cells as 4 if the rectangle is large enough
    min_block_area = 20
    
    result = g.copy()
    
    for color in np.unique(g):
        mask = (g == color).astype(int)
        heights = np.zeros(w, dtype=int)
        best_area = 0
        best_rect = None
        
        for r in range(h):
            heights = np.where(mask[r] == 1, heights + 1, 0)
            stack = []
            for c in range(w + 1):
                h_cur = heights[c] if c < w else 0
                start = c
                while stack and stack[-1][1] > h_cur:
                    sc, sh = stack.pop()
                    area = sh * (c - sc)
                    if area > best_area:
                        best_area = area
                        best_rect = (r-sh+1, r, sc, c-1)
                    start = sc
                stack.append((start, h_cur))
        
        if best_rect and best_area >= min_block_area:
            r1, r2, c1, c2 = best_rect
            result[r1:r2+1, c1:c2+1] = 4
    
    return result.tolist()
