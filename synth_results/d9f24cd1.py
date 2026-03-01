def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    for start_col in range(cols):
        if grid[rows-1][start_col] == 2:
            col = start_col
            r = rows - 1
            while r >= 0:
                if grid[r][col] == 5:
                    col += 1
                    if col >= cols: break
                    result[r+1][col] = 2
                    result[r][col] = 2
                    r -= 1
                    continue
                elif grid[r][col] == 0:
                    result[r][col] = 2
                r -= 1
    return result
