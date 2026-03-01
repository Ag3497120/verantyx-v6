def transform(grid):
    import copy
    result = copy.deepcopy(grid)
    height = len(grid)
    width = len(grid[0])
    
    # Find all positions with value 1
    ones = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1:
                ones.append((r, c))
    
    # For each 1, draw the cross lines
    for r, c in ones:
        # Draw vertical line
        for row in range(height):
            result[row][c] = 1
        # Draw horizontal line
        for col in range(width):
            result[r][col] = 1
    
    # Then mark centers and diagonals
    for r, c in ones:
        # Mark center as 2
        result[r][c] = 2
        
        # Mark diagonal cells as 3
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if result[nr][nc] != 2:
                    result[nr][nc] = 3
    
    return result
