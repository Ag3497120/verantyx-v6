def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find background color (most common)
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    non_bg = [(r, c, g[r, c]) for r in range(rows) for c in range(cols) if g[r, c] != bg]
    if not non_bg:
        return grid
    
    values = set(v for _, _, v in non_bg)
    
    # Find the "rectangle" (large connected block of non-bg cells)
    from scipy.ndimage import label
    nz_mask = (g != bg).astype(int)
    labeled, num = label(nz_mask)
    comp_sizes = [(np.sum(labeled == i), i) for i in range(1, num+1)]
    comp_sizes.sort(reverse=True)
    
    if not comp_sizes:
        return grid
    
    rect_comp = comp_sizes[0][1]
    rect_cells = np.argwhere(labeled == rect_comp)
    r_min, c_min = rect_cells.min(axis=0)
    r_max, c_max = rect_cells.max(axis=0)
    
    # Signal cells are isolated (small components)
    for sz, comp in comp_sizes[1:]:
        if sz > 4:
            continue  # not a signal
        signal_cells = np.argwhere(labeled == comp)
        for sr, sc in signal_cells:
            sv = g[sr, sc]
            # Draw line from signal to rectangle edge
            # If signal is above/below the rectangle (same column range)
            if c_min <= sc <= c_max:
                # Vertical signal
                if sr < r_min:
                    for r in range(sr, r_min):
                        result[r, sc] = sv
                elif sr > r_max:
                    for r in range(r_max+1, sr+1):
                        result[r, sc] = sv
            elif r_min <= sr <= r_max:
                # Horizontal signal
                if sc < c_min:
                    for c in range(sc, c_min):
                        result[sr, c] = sv
                elif sc > c_max:
                    for c in range(c_max+1, sc+1):
                        result[sr, c] = sv
    
    return result.tolist()
