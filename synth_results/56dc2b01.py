
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    # Find 3-shape positions
    shape3 = list(zip(*np.where(g == 3)))
    if not shape3:
        return g.tolist()
    # Find 2-bar: a row or col entirely of 2s (except 0s)
    bar_row = bar_col = None
    for r in range(h):
        if any(g[r,c] == 2 for c in range(w)) and all(g[r,c] in (0,2) for c in range(w)):
            bar_row = r; break
    for c in range(w):
        if any(g[r,c] == 2 for r in range(h)) and all(g[r,c] in (0,2) for r in range(h)):
            bar_col = c; break
    
    out = np.zeros_like(g)
    # Preserve 2-bar
    out[g == 2] = 2
    
    if bar_row is not None:
        # Vertical fall toward the 2-bar
        rows3 = [r for r,c in shape3]
        min_r3, max_r3 = min(rows3), max(rows3)
        shape_h = max_r3 - min_r3 + 1
        if bar_row > max_r3:
            # Fall down: new bottom = bar_row - 1, new top = bar_row - shape_h
            new_top = bar_row - shape_h
            shift = new_top - min_r3
            # Place 8-bar at new_top - 1
            out[new_top - 1, :] = 8
        else:
            # Fall up: new top = bar_row + 1, new bottom = bar_row + shape_h
            new_top = bar_row + 1
            shift = new_top - min_r3
            # Place 8-bar at new_top + shape_h (below shape)
            out[new_top + shape_h, :] = 8
        for r, c in shape3:
            out[r + shift, c] = 3
    elif bar_col is not None:
        # Horizontal fall toward the 2-bar
        cols3 = [c for r,c in shape3]
        min_c3, max_c3 = min(cols3), max(cols3)
        shape_w = max_c3 - min_c3 + 1
        if bar_col > max_c3:
            # Fall right: new right = bar_col - 1, new left = bar_col - shape_w
            new_left = bar_col - shape_w
            shift = new_left - min_c3
            # 8-bar at new_left - 1
            out[:, new_left - 1] = 8
        else:
            # Fall left: new left = bar_col + 1, new right = bar_col + shape_w
            new_left = bar_col + 1
            shift = new_left - min_c3
            # 8-bar at new_left + shape_w
            out[:, new_left + shape_w] = 8
        for r, c in shape3:
            out[r, c + shift] = 3
    
    return out.tolist()
