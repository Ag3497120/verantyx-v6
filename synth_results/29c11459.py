def transform(grid):
    g = [list(r) for r in grid]
    rows, cols = len(g), len(g[0])
    for r in range(rows):
        row = g[r]
        nz = [(c, v) for c, v in enumerate(row) if v != 0]
        if len(nz) == 2:
            (c1, v1), (c2, v2) = nz
            if c1 > c2:
                c1, v1, c2, v2 = c2, v2, c1, v1
            mid = (c1 + c2) // 2
            for c in range(c1, mid):
                g[r][c] = v1
            g[r][mid] = 5
            for c in range(mid+1, c2+1):
                g[r][c] = v2
    return g
