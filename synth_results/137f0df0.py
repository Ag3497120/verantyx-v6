def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find 5-rows and 5-cols
    five_rows = set(r for r in range(rows) if any(g[r, c] == 5 for c in range(cols)))
    five_cols = set(c for c in range(cols) if any(g[r, c] == 5 for r in range(rows)))
    
    if not five_rows or not five_cols:
        return grid
    
    min_5r, max_5r = min(five_rows), max(five_rows)
    min_5c, max_5c = min(five_cols), max(five_cols)
    
    # gap: between min and max 5-position, not in five set
    # tail: outside min-max range of 5-positions
    
    for r in range(rows):
        for c in range(cols):
            if g[r, c] != 0:
                continue  # keep 5s as is
            
            r_in_5 = r in five_rows
            r_gap = (not r_in_5) and (min_5r <= r <= max_5r)
            r_tail = r > max_5r or r < min_5r
            
            c_in_5 = c in five_cols
            c_gap = (not c_in_5) and (min_5c <= c <= max_5c)
            c_tail = c > max_5c or c < min_5c
            
            # Determine fill value
            if r_tail and c_gap:
                result[r, c] = 1
            elif c_tail and r_gap:
                result[r, c] = 1
            elif c_gap or r_gap:
                # Either gap-row or gap-col (and the other is in-5 or gap)
                if not r_tail and not c_tail:
                    result[r, c] = 2
            # else: tail-tail or in5-tail → stays 0
    
    return result.tolist()
