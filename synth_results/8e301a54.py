def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    bg = int(np.bincount(g.flatten()).argmax())
    out = np.full_like(g, bg)
    # Find unique non-bg values
    unique_vals = [v for v in np.unique(g) if v != bg]
    for val in unique_vals:
        positions = np.argwhere(g == val)
        n = len(positions)
        # Move each cell down by n positions
        for pos in positions:
            new_r = int(pos[0]) + n
            new_c = int(pos[1])
            if 0 <= new_r < H:
                out[new_r, new_c] = val
    return out.tolist()
