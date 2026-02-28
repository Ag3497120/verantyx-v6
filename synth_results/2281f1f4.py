def transform(grid):
    # grid is list of lists of ints, 10x10
    rows = len(grid)
    cols = len(grid[0])
    
    # Columns with a 5 in the first row
    cols_with_5_in_first = [c for c in range(cols) if grid[0][c] == 5]
    
    # Rows with a 5 in the last column
    rows_with_5_in_last = [r for r in range(rows) if grid[r][cols-1] == 5]
    
    # Create a copy to modify
    result = [row[:] for row in grid]
    
    for r in rows_with_5_in_last:
        for c in cols_with_5_in_first:
            if result[r][c] == 0:
                result[r][c] = 2
    
    return result