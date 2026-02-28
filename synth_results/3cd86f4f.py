def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    H = rows
    W = cols
    new_width = W + H - 1
    result = []
    for i, row in enumerate(grid):
        new_row = [0]*(H-1-i) + list(row) + [0]*i
        result.append(new_row)
    return result
