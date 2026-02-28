def transform(grid):
    import copy
    result = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                # fill right alternating v, 5, v, 5...
                for dc in range(cols - c):
                    result[r][c + dc] = v if dc % 2 == 0 else 5
                break  # only first non-zero per row? Let's check
    return result
