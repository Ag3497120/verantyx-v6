def transform(grid):
    N = len(grid)
    C = grid[0][0]
    period = N + 1
    result = []
    for r in range(15):
        row = []
        for c in range(15):
            if r % period == N or c % period == N:
                row.append(C)
            else:
                row.append(0)
        result.append(row)
    return result
