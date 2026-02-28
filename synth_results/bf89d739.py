def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    twos = [(r,c) for r in range(rows) for c in range(cols) if g[r][c]==2]
    if len(twos) < 2: return g
    # Find pair sharing row or column
    primary = None
    for i in range(len(twos)):
        for j in range(i+1, len(twos)):
            r1,c1 = twos[i]; r2,c2 = twos[j]
            if r1==r2 or c1==c2:
                primary = (twos[i], twos[j])
                break
        if primary: break
    if primary is None:
        # No pair in same row/col - just connect first two with L-path
        r1,c1 = twos[0]; r2,c2 = twos[1]
        for r in range(min(r1,r2)+1, max(r1,r2)):
            g[r][c1] = 3
        for c in range(min(c1,c2)+1, max(c1,c2)):
            g[r2][c] = 3
        return g
    (r1,c1),(r2,c2) = primary
    # Draw primary line
    if r1==r2:
        for c in range(min(c1,c2)+1, max(c1,c2)):
            g[r1][c] = 3
        # Others connect via vertical
        others = [t for t in twos if t not in [(r1,c1),(r2,c2)]]
        for ro,co in others:
            # Connect at column co, row r1
            if ro < r1:
                for rr in range(ro+1, r1): g[rr][co] = 3
            else:
                for rr in range(r1+1, ro): g[rr][co] = 3
    else:
        for r in range(min(r1,r2)+1, max(r1,r2)):
            g[r][c1] = 3
        others = [t for t in twos if t not in [(r1,c1),(r2,c2)]]
        for ro,co in others:
            if co < c1:
                for cc in range(co+1, c1): g[ro][cc] = 3
            else:
                for cc in range(c1+1, co): g[ro][cc] = 3
    return g
