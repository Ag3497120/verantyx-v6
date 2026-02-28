def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    H, W = g.shape
    labeled, n = label(g > 0)
    out = g.copy()
    for i in range(1, n+1):
        comp_mask = labeled == i
        rows = np.where(comp_mask.any(axis=1))[0]
        cols = np.where(comp_mask.any(axis=0))[0]
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        interior_h = r2 - r1 - 1
        interior_w = c2 - c1 - 1
        if interior_h <= 0 or interior_w <= 0:
            continue
        # Odd interior size → 7, even → 2
        fill_color = 7 if interior_h % 2 == 1 else 2
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                if g[r, c] == 0:
                    out[r, c] = fill_color
    return out.tolist()
