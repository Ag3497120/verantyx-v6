def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    # Find divider column (all zeros)
    div_col = None
    for c in range(C):
        if np.all(g[:, c] == 0):
            div_col = c
            break
    if div_col is None:
        return grid
    
    left = g[:, :div_col]
    right = g[:, div_col:]
    bg = left[0, 0]
    
    div_rows = [r for r in range(R) if np.all(left[r] == bg)]
    
    sections = []
    prev = -1
    for dr in div_rows:
        if dr > prev + 1:
            sections.append((prev + 1, dr - 1))
        prev = dr
    if prev < R - 1:
        sections.append((prev + 1, R - 1))
    
    def get_shape(cells):
        if not cells:
            return frozenset()
        arr = np.array(cells)
        arr -= arr.min(axis=0)
        return frozenset(map(tuple, arr.tolist()))
    
    holes = []
    patterns = []
    for r1, r2 in sections:
        h_cells = [(r - r1, c) for r in range(r1, r2 + 1)
                   for c in range(div_col) if left[r, c] == 0]
        holes.append(h_cells)
        
        p_cells = [(r - r1, c) for r in range(r1, r2 + 1)
                   for c in range(right.shape[1]) if right[r, c] != 0]
        color = right[p_cells[0][0] + r1, p_cells[0][1]] if p_cells else 0
        patterns.append((p_cells, int(color)))
    
    out_left = left.copy()
    used = set()
    for i, h_cells in enumerate(holes):
        if not h_cells:
            continue
        h_shape = get_shape(h_cells)
        for j, (p_cells, color) in enumerate(patterns):
            if j in used:
                continue
            p_shape = get_shape(p_cells)
            if p_shape == h_shape:
                r1, r2 = sections[i]
                for dr, c in h_cells:
                    out_left[r1 + dr, c] = color
                used.add(j)
                break
    
    return out_left.tolist()
