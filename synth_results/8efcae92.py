def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    labeled, n = label(g > 0)
    best_comp = None; max_2s = -1
    for i in range(1, n+1):
        mask = labeled == i
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        region = g[r1:r2+1, c1:c2+1]
        n2 = int(np.sum(region == 2))
        if n2 > max_2s:
            max_2s = n2; best_comp = region
    return best_comp.tolist() if best_comp is not None else grid
