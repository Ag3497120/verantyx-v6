def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    cells5 = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==5]
    if not cells5: return result
    minr = min(r for r,c in cells5); maxr = max(r for r,c in cells5)
    minc = min(c for r,c in cells5); maxc = max(c for r,c in cells5)
    for r in range(minr+1, maxr):
        for c in range(minc+1, maxc):
            result[r][c] = 8
    cells5_set = set(cells5)
    top_missing = [c for c in range(minc, maxc+1) if (minr,c) not in cells5_set]
    bot_missing = [c for c in range(minc, maxc+1) if (maxr,c) not in cells5_set]
    left_missing = [r for r in range(minr, maxr+1) if (r,minc) not in cells5_set]
    right_missing = [r for r in range(minr, maxr+1) if (r,maxc) not in cells5_set]
    if top_missing:
        for c in top_missing:
            result[minr][c] = 8
            for r in range(0, minr): result[r][c] = 8
    if bot_missing:
        for c in bot_missing:
            result[maxr][c] = 8
            for r in range(maxr+1, rows): result[r][c] = 8
    if left_missing:
        for r in left_missing:
            result[r][minc] = 8
            for c in range(0, minc): result[r][c] = 8
    if right_missing:
        for r in right_missing:
            result[r][maxc] = 8
            for c in range(maxc+1, cols): result[r][c] = 8
    return result
