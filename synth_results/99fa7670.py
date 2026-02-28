def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find all non-zero pixels
    pixels = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    
    for pr, pc, pv in pixels:
        # Shoot right from this pixel to right edge
        for c in range(pc, cols):
            result[pr][c] = pv
        # Shoot down from right edge
        for r in range(pr, rows):
            result[r][cols-1] = pv
    
    return result
