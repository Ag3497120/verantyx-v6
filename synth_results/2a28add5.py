def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = 7
    result = [[bg]*cols for _ in range(rows)]
    
    for r in range(rows):
        row = grid[r]
        # Find non-bg values and their columns
        nz = [(c, row[c]) for c in range(cols) if row[c] != bg]
        if not nz:
            continue
        # Find column of value 6
        six_cols = [c for c, v in nz if v == 6]
        if not six_cols:
            continue
        six_col = six_cols[0]
        
        n = len(nz)  # total non-bg count = width of 8-block
        # Count non-bg to left of 6
        left_count = sum(1 for c, v in nz if c < six_col)
        
        start = six_col - left_count
        end = six_col + (n - left_count - 1)
        
        for c in range(max(0, start), min(cols, end + 1)):
            result[r][c] = 8
    
    return result
