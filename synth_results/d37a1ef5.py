def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    cells2 = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==2]
    if not cells2: return result
    minr=min(r for r,c in cells2); maxr=max(r for r,c in cells2)
    minc=min(c for r,c in cells2); maxc=max(c for r,c in cells2)
    for r in range(minr+1, maxr):
        for c in range(minc+1, maxc):
            if grid[r][c] == 0: result[r][c] = 2
    cells5 = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==5]
    if cells5:
        r5_min=min(r for r,c in cells5); r5_max=max(r for r,c in cells5)
        c5_min=min(c for r,c in cells5); c5_max=max(c for r,c in cells5)
        for r in range(r5_min, r5_max+1):
            for c in range(c5_min, c5_max+1):
                if grid[r][c] != 5: result[r][c] = 0
    return result
