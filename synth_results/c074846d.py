def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    anchor = None
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == 5: anchor = (r,c)
    if not anchor: return grid
    ar, ac = anchor
    twos = [(r,c) for r in range(rows) for c in range(cols) if g[r][c] == 2]
    out = [list(row) for row in g]
    for r,c in twos:
        out[r][c] = 3  # original 2 → 3
        # rotate 90° CW around anchor: dr,dc → dc,-dr
        dr = r - ar; dc = c - ac
        nr = ar + dc; nc = ac - dr
        if 0<=nr<rows and 0<=nc<cols:
            out[nr][nc] = 2
    return out
