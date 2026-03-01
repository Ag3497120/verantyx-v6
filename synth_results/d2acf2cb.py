def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    def swap(v):
        if v == 0: return 8
        if v == 8: return 0
        if v == 6: return 7
        if v == 7: return 6
        return v
    for r in range(rows):
        if grid[r][0] == 4 and grid[r][cols-1] == 4:
            for c in range(1, cols-1):
                result[r][c] = swap(grid[r][c])
    for c in range(cols):
        if grid[0][c] == 4 and grid[rows-1][c] == 4:
            for r in range(1, rows-1):
                result[r][c] = swap(grid[r][c])
    return result
