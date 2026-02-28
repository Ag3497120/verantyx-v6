def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    H, W = g.shape
    # Find wall color (most common non-zero value)
    vals = g[g > 0].flatten().tolist()
    if not vals:
        return grid
    wall_color = Counter(vals).most_common(1)[0][0]
    # Find separator rows/cols (rows/cols that are mostly wall_color)
    sep_rows = sorted([r for r in range(H) if np.sum(g[r] == wall_color) > W // 3])
    sep_cols = sorted([c for c in range(W) if np.sum(g[:, c] == wall_color) > H // 3])
    # Build bands
    def bands(seps, size):
        b = []
        prev = 0
        for s in seps:
            if prev < s:
                b.append((prev, s-1))
            prev = s + 1
        if prev < size:
            b.append((prev, size-1))
        return b
    rbands = bands(sep_rows, H)
    cbands = bands(sep_cols, W)
    out = g.copy()
    # Fill non-wall cells with wall_color first, then recolor
    out[g == 0] = 3  # default
    for ri, (r1, r2) in enumerate(rbands):
        for ci, (c1, c2) in enumerate(cbands):
            gaps = False
            if ri > 0:
                sep_r = sep_rows[ri-1]
                if 0 in g[sep_r, c1:c2+1]: gaps = True
            if ri < len(rbands)-1:
                sep_r = sep_rows[ri]
                if 0 in g[sep_r, c1:c2+1]: gaps = True
            if ci > 0:
                sep_c = sep_cols[ci-1]
                if 0 in g[r1:r2+1, sep_c]: gaps = True
            if ci < len(cbands)-1:
                sep_c = sep_cols[ci]
                if 0 in g[r1:r2+1, sep_c]: gaps = True
            color = 4 if gaps else 3
            # Fill cell region (non-wall cells)
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if g[r, c] == 0:
                        out[r, c] = color
    # Fill gaps in separator rows/cols with 4
    for sr in sep_rows:
        for c in range(W):
            if g[sr, c] == 0:
                out[sr, c] = 4
    for sc in sep_cols:
        for r in range(H):
            if g[r, sc] == 0:
                out[r, sc] = 4
    return out.tolist()
