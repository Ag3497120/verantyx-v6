def transform(grid):
    rows, cols = len(grid), len(grid[0])
    # Find wall: a full row or column of same non-zero color
    wall_col = None
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        non_zero = [v for v in col_vals if v != 0]
        if len(non_zero) == rows and len(set(non_zero)) == 1:
            wall_col = c
            break
    wall_row = None
    for r in range(rows):
        row_vals = grid[r]
        non_zero = [v for v in row_vals if v != 0]
        if len(non_zero) == cols and len(set(non_zero)) == 1:
            wall_row = r
            break
    result = [row[:] for row in grid]
    if wall_col is not None:
        if wall_col > cols // 2:
            for r in range(rows):
                cur = 0
                for c in range(cols):
                    if c == wall_col: break
                    if result[r][c] != 0:
                        cur = result[r][c]
                    elif cur != 0:
                        result[r][c] = cur
        else:
            for r in range(rows):
                cur = 0
                for c in range(cols-1, -1, -1):
                    if c == wall_col: break
                    if result[r][c] != 0:
                        cur = result[r][c]
                    elif cur != 0:
                        result[r][c] = cur
    elif wall_row is not None:
        if wall_row > rows // 2:
            for c in range(cols):
                cur = 0
                for r in range(rows):
                    if r == wall_row: break
                    if result[r][c] != 0:
                        cur = result[r][c]
                    elif cur != 0:
                        result[r][c] = cur
        else:
            for c in range(cols):
                cur = 0
                for r in range(rows-1, -1, -1):
                    if r == wall_row: break
                    if result[r][c] != 0:
                        cur = result[r][c]
                    elif cur != 0:
                        result[r][c] = cur
    return result
