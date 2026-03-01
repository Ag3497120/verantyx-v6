def transform(grid):
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    # For each non-zero in row 0, create zigzag pattern
    for col in range(w):
        if grid[0][col] != 0:
            color = grid[0][col]
            
            # Create zigzag pattern downward
            for row in range(1, h):
                if row % 2 == 1:  # Odd rows: shift left
                    if col > 0:
                        grid[row][col - 1] = color
                    if col < w - 1:
                        grid[row][col + 1] = color
                else:  # Even rows: same column
                    grid[row][col] = color
    
    return grid
