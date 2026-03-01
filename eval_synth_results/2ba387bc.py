def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from collections import Counter
    inp = np.array(grid)
    non_zero = (inp != 0).astype(int)
    labeled, n = label(non_zero)
    hollows = []
    solids = []
    for i in range(1, n+1):
        mask = labeled == i
        rows_m, cols_m = np.where(mask)
        r0, r1, c0, c1 = rows_m.min(), rows_m.max(), cols_m.min(), cols_m.max()
        region = inp[r0:r1+1, c0:c1+1].copy()
        val = int(Counter(inp[mask].tolist()).most_common()[0][0])
        is_solid = bool(np.all(region == val))
        if is_solid:
            solids.append((r0, val, region))
        else:
            hollows.append((r0, val, region))
    hollows.sort(key=lambda x: x[0])
    solids.sort(key=lambda x: x[0])
    n_pairs = max(len(hollows), len(solids))
    h = hollows[0][2].shape[0] if hollows else (solids[0][2].shape[0] if solids else 4)
    w = hollows[0][2].shape[1] if hollows else (solids[0][2].shape[1] if solids else 4)
    out = np.zeros((n_pairs * h, w * 2), dtype=int)
    for i in range(n_pairs):
        row_start = i * h
        if i < len(hollows):
            out[row_start:row_start+h, :w] = hollows[i][2]
        if i < len(solids):
            out[row_start:row_start+h, w:] = solids[i][2]
    return out.tolist()
