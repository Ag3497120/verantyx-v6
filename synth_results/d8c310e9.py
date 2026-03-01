def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    bottom_row = -1
    for r in range(rows-1, -1, -1):
        if any(grid[r][c] != 0 for c in range(cols)):
            bottom_row = r; break
    if bottom_row == -1: return result
    row = grid[bottom_row]
    P = None
    for p in range(1, cols):
        valid = True
        for c in range(cols):
            if c + p < cols and row[c+p] != 0 and row[c] != 0 and row[c+p] != row[c]:
                valid = False; break
            if c + p < cols and row[c+p] != 0 and row[c] == 0:
                valid = False; break
        if valid and row[p] != 0:
            P = p; break
    if P is None: return result
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                source_c = c % P
                if grid[r][source_c] != 0:
                    result[r][c] = grid[r][source_c]
    return result
