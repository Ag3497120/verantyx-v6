def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find 5 position (marker)
    five_col = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_col = c
                break
        if five_col is not None:
            break
    
    if five_col is None:
        return grid
    
    result = [row[:] for row in grid]
    
    for r in range(rows):
        # Find horizontal line in this row (ignoring 5)
        line = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0 and grid[r][c] != 5]
        if not line:
            continue
        # Line length
        length = len(line)
        # Output value = (length-1)^2 % 10
        output_val = ((length - 1) ** 2) % 10
        result[r][five_col] = output_val
    
    return result
