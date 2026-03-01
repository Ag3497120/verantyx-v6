def transform(grid):
    R = len(grid)
    C = len(grid[0])
    min_r, max_r, min_c, max_c = R, 0, C, 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 8:
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c
    result = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            found = None
            for nr, nc in [(r, 31-c), (31-r, c), (31-r, 31-c),
                           (c, r), (31-c, r), (c, 31-r), (31-c, 31-r)]:
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != 8:
                    found = grid[nr][nc]
                    break
            row.append(found)
        result.append(row)
    return result
