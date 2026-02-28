def transform(grid):
    H = len(grid)
    s = H // 3
    sec0 = grid[0:s]       # color 1
    sec1 = grid[s:2*s]     # color 8
    sec2 = grid[2*s:3*s]   # color 6
    result = []
    for r in range(s):
        row = []
        for c in range(len(grid[0])):
            v6 = sec2[r][c]
            v1 = sec0[r][c]
            v8 = sec1[r][c]
            if v6 != 0:
                row.append(v6)
            elif v1 != 0:
                row.append(v1)
            elif v8 != 0:
                row.append(v8)
            else:
                row.append(0)
        result.append(row)
    return result
