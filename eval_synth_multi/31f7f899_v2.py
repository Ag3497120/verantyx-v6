def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    bg = 8
    
    # Find the horizontal line row
    h_row = max(range(rows), key=lambda r: int(np.sum(g[r] != bg)))
    
    from collections import Counter
    h_vals = [int(g[h_row, c]) for c in range(cols) if g[h_row, c] != bg]
    h_color = Counter(h_vals).most_common(1)[0][0]
    
    # Find vertical colored bars
    v_lines = []
    for c in range(cols):
        color = int(g[h_row, c])
        if color == bg or color == h_color:
            continue
        above = 0
        for r in range(h_row - 1, -1, -1):
            if int(g[r, c]) == color:
                above += 1
            else:
                break
        below = 0
        for r in range(h_row + 1, rows):
            if int(g[r, c]) == color:
                below += 1
            else:
                break
        v_lines.append((c, color, above, below))
    
    v_lines.sort(key=lambda x: x[0])
    
    # Sort above and below extents independently (ascending, left to right)
    above_sorted = sorted([v[2] for v in v_lines])
    below_sorted = sorted([v[3] for v in v_lines])
    
    out = np.full_like(g, bg)
    out[h_row] = g[h_row].copy()
    
    for i, (c, color, _, _) in enumerate(v_lines):
        new_above = above_sorted[i]
        new_below = below_sorted[i]
        out[h_row, c] = color
        for d in range(1, new_above + 1):
            if h_row - d >= 0:
                out[h_row - d, c] = color
        for d in range(1, new_below + 1):
            if h_row + d < rows:
                out[h_row + d, c] = color
    
    return out.tolist()
