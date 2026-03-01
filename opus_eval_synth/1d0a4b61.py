def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find horizontal period
    for px in range(1, cols):
        ok = True
        for r in range(rows):
            for c in range(cols):
                c2 = c % px
                if grid[r][c] != 0 and grid[r][c2] != 0 and grid[r][c] != grid[r][c2]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break

    # Find vertical period
    for py in range(1, rows):
        ok = True
        for r in range(rows):
            r2 = r % py
            for c in range(cols):
                if grid[r][c] != 0 and grid[r2][c] != 0 and grid[r][c] != grid[r2][c]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break

    # Build tile from non-zero values
    tile = [[None] * px for _ in range(py)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                tr, tc = r % py, c % px
                if tile[tr][tc] is None:
                    tile[tr][tc] = grid[r][c]

    # Fill grid using tile
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(tile[r % py][c % px])
        out.append(row)
    return out
