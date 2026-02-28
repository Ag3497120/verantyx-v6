def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    twos = [(r,c) for r in range(rows) for c in range(cols) if g[r][c] == 2]
    if len(twos) < 2: return g
    (r1,c1),(r2,c2) = twos[0], twos[1]
    # Draw line from (r1,c1) to (r2,c2) using Bresenham or step
    dr = r2-r1; dc = c2-c1
    steps = max(abs(dr), abs(dc))
    if steps == 0: return g
    for i in range(steps+1):
        r = r1 + round(i*dr/steps)
        c = c1 + round(i*dc/steps)
        if 0<=r<rows and 0<=c<cols:
            if g[r][c] == 1: g[r][c] = 3
            elif g[r][c] == 0: g[r][c] = 2
    return g
