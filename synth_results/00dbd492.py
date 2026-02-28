def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Interior size (width=height) -> fill color
    # Derived from training: 3->8, 5->4, 7->3
    color_map = {3: 8, 5: 4, 7: 3, 1: 9, 9: 2, 11: 1}
    
    # Find all rectangular frames made of 2s
    # Look for rows/cols that form a border
    visited = np.zeros_like(g, dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if g[r,c] == 2 and not visited[r,c]:
                # Try to find rectangular frame starting here
                # Find the extent of this frame
                # Look right for top edge
                if r+1 < rows and c+1 < cols:
                    # Find bottom-right corner
                    # Find right edge: scan right from (r,c)
                    c2 = c
                    while c2+1 < cols and g[r, c2+1] == 2:
                        c2 += 1
                    # Find bottom edge: scan down from (r,c)
                    r2 = r
                    while r2+1 < rows and g[r2+1, c] == 2:
                        r2 += 1
                    
                    if r2 > r and c2 > c:
                        # Check if it's a valid rectangle border
                        # Top and bottom rows all 2
                        top_ok = all(g[r,c:c2+1] == 2)
                        bot_ok = all(g[r2,c:c2+1] == 2)
                        left_ok = all(g[r:r2+1,c] == 2)
                        right_ok = all(g[r:r2+1,c2] == 2)
                        
                        if top_ok and bot_ok and left_ok and right_ok:
                            # Interior region
                            inner_h = r2 - r - 1
                            inner_w = c2 - c - 1
                            dim = min(inner_h, inner_w)
                            fill = color_map.get(dim, dim)
                            
                            # Fill interior (keep any 2s inside as 2)
                            for ir in range(r+1, r2):
                                for ic in range(c+1, c2):
                                    if g[ir, ic] == 0:
                                        result[ir, ic] = fill
                            
                            visited[r:r2+1, c:c2+1] = True
    
    return result.tolist()
