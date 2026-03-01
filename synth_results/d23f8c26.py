def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    mid = cols // 2
    for r in range(rows): result[r][mid] = grid[r][mid]
    return result
