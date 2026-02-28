def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    nz = np.argwhere(g != 0)
    if len(nz) == 0: return grid
    start_r, start_c = int(nz[0][0]), int(nz[0][1])
    val = int(g[start_r, start_c])
    start_side = 'bottom' if start_r == H-1 else 'top'
    # Draw vertical rails at start_c, start_c+2, ...
    rails = []
    c = start_c
    while c < W:
        rails.append(c)
        out[:, c] = val
        c += 2
    # For each gap (including potential gap after last rail), place 5 at alternating ends
    first_gap_top = (start_side == 'bottom')
    gap_idx = 0
    # Gaps between adjacent rails
    for i in range(len(rails) - 1):
        gap_col = rails[i] + 1
        if gap_col < W:
            use_top = first_gap_top if gap_idx % 2 == 0 else not first_gap_top
            out[0 if use_top else H-1, gap_col] = 5
            gap_idx += 1
    # Gap after last rail (if within bounds)
    after_col = rails[-1] + 1
    if after_col < W:
        use_top = first_gap_top if gap_idx % 2 == 0 else not first_gap_top
        out[0 if use_top else H-1, after_col] = 5
    return out.tolist()
