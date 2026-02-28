def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    # Find 5 position
    five_pos = np.argwhere(g == 5)
    if len(five_pos) == 0:
        return grid
    fr, fc = five_pos[0]
    # Find template: all non-zero, non-5 cells
    template_cells = np.argwhere((g != 0) & (g != 5))
    if len(template_cells) == 0:
        return grid
    tr1 = template_cells[:, 0].min()
    tr2 = template_cells[:, 0].max()
    tc1 = template_cells[:, 1].min()
    tc2 = template_cells[:, 1].max()
    th = tr2 - tr1 + 1
    tw = tc2 - tc1 + 1
    # Template center
    tcr = tr1 + th // 2
    tcc = tc1 + tw // 2
    # Copy offset
    dr = fr - tcr
    dc = fc - tcc
    # Place copy
    for r, c in template_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = g[r, c]
    # Replace the 5 with the template value at the 5's position (relative to copy)
    out[fr, fc] = g[fr - dr, fc - dc]
    return out.tolist()
