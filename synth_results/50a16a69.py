def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    h, w = g.shape
    # Find bg color from last row/col
    last_row = g[-1]
    last_col = g[:,-1]
    if len(set(last_row.tolist())) == 1 and len(set(last_col.tolist())) == 1 and last_row[0] == last_col[0]:
        bg = int(last_row[0])
    else:
        border = list(g[0]) + list(g[-1]) + list(g[:,0]) + list(g[:,-1])
        bg = Counter(border).most_common(1)[0][0]
    # Find non-bg region
    non_bg_rows = [r for r in range(h) if any(g[r,c] != bg for c in range(w))]
    non_bg_cols = [c for c in range(w) if any(g[r,c] != bg for r in range(h))]
    r0, r1 = non_bg_rows[0], non_bg_rows[-1]+1
    c0, c1 = non_bg_cols[0], non_bg_cols[-1]+1
    pattern = g[r0:r1, c0:c1]
    ph, pw = pattern.shape
    def find_period(arr):
        n = len(arr)
        for p in range(1, n+1):
            tile = arr[:p]
            reconstructed = np.tile(tile, (n // p) + 1)[:n]
            if np.array_equal(reconstructed, arr):
                return p
        return n
    rp = find_period(pattern[:,0])
    cp = find_period(pattern[0,:])
    tile = pattern[:rp, :cp]
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r,c] = tile[r % rp, (c+1) % cp]
    return out.tolist()
