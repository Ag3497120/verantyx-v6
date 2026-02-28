def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    
    # Find separator rows and cols (that are uniform in a value)
    sep_rows = []
    sep_cols = []
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1:
            sep_rows.append((r, grid[r][0]))
    for c in range(cols):
        col_vals = set(grid[r][c] for r in range(rows))
        if len(col_vals) == 1:
            sep_cols.append((c, grid[0][c]))
    
    if not sep_rows or not sep_cols:
        return grid
    
    sep_row, sep_row_val = sep_rows[0]
    sep_col, sep_col_val = sep_cols[0]
    
    # Find dominant color in each quadrant
    def dominant(cells):
        c = Counter(cells)
        return c.most_common(1)[0][0]
    
    tl = [grid[r][c] for r in range(0, sep_row) for c in range(0, sep_col) if grid[r][c] != sep_row_val and grid[r][c] != sep_col_val]
    tr = [grid[r][c] for r in range(0, sep_row) for c in range(sep_col+1, cols) if grid[r][c] != sep_row_val and grid[r][c] != sep_col_val]
    bl = [grid[r][c] for r in range(sep_row+1, rows) for c in range(0, sep_col) if grid[r][c] != sep_row_val and grid[r][c] != sep_col_val]
    br = [grid[r][c] for r in range(sep_row+1, rows) for c in range(sep_col+1, cols) if grid[r][c] != sep_row_val and grid[r][c] != sep_col_val]
    
    sv = sep_row_val  # separator value
    return [
        [dominant(tl), sv, dominant(tr)],
        [sv, sv, sv],
        [dominant(bl), sv, dominant(br)]
    ]
