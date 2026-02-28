def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the rectangle defined by 5s
    five_rows = set()
    five_cols = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_rows.add(r)
                five_cols.add(c)
    
    if not five_rows or not five_cols:
        return grid
    
    r1, r2 = min(five_rows), max(five_rows)
    c1, c2 = min(five_cols), max(five_cols)
    
    # Include one row above and below (the 8 border rows)
    r1_out = r1 - 1
    r2_out = r2 + 1
    
    result = []
    for r in range(r1_out, r2_out + 1):
        if 0 <= r < rows:
            row = [grid[r][c] for c in range(c1, c2+1)]
            result.append(row)
    
    return result
