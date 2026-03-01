def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    for val in range(1, 10):
        cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==val]
        if not cells: continue
        minr = min(r for r,c in cells); maxr = max(r for r,c in cells)
        minc = min(c for r,c in cells); maxc = max(c for r,c in cells)
        for gap_col in range(minc+1, maxc):
            if not any(grid[r][gap_col]==val for r in range(minr, maxr+1)):
                for r in range(rows):
                    if grid[r][gap_col] == 0: result[r][gap_col] = 3
                return result
        for gap_row in range(minr+1, maxr):
            if not any(grid[gap_row][c]==val for c in range(minc, maxc+1)):
                for c in range(cols):
                    if grid[gap_row][c] == 0: result[gap_row][c] = 3
                return result
    return result
