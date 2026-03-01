def transform(grid):
    import copy
    result = copy.deepcopy(grid)
    height = len(grid)
    width = len(grid[0])
    
    # Find all positions with value 4
    fours = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 4:
                fours.append((r, c))
    
    # For each 4, convert adjacent 5s to 2s
    for r, c in fours:
        # Check all 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if result[nr][nc] == 5:
                    result[nr][nc] = 2
    
    return result
