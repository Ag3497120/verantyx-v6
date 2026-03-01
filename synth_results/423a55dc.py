
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all non-zero cells
    nonzero = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    if not nonzero:
        return grid
    
    # Find bounding box
    r_min = min(r for r,c in nonzero)
    r_max = max(r for r,c in nonzero)
    
    # Each row shifts left by (r_max - r)
    result = [row[:] for row in grid]
    for r,c in nonzero:
        shift = r_max - r
        new_c = c - shift
        result[r][c] = 0
        if 0 <= new_c < cols:
            result[r][new_c] = grid[r][c]
    
    return result
