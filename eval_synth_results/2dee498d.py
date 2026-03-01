def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    for p in range(1, cols + 1):
        if cols % p != 0:
            continue
        valid = True
        for r in range(rows):
            base = grid[r][:p]
            base_rev = base[::-1]
            for k in range(1, cols // p):
                chunk = grid[r][k*p:(k+1)*p]
                if chunk != base and chunk != base_rev:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [row[:p] for row in grid]
    return grid
