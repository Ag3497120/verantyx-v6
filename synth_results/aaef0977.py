def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find cursor (single non-bg cell)
    cursor = [(r,c,grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c]!=bg]
    if not cursor:
        return grid
    r0, c0, cv = cursor[0]
    
    # Fixed cycle (excluding bg=7): [3,4,0,5,2,8,9,6,1]
    cycle = [3,4,0,5,2,8,9,6,1]
    start_idx = cycle.index(cv)
    
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            dist = abs(r-r0) + abs(c-c0)
            result[r][c] = cycle[(start_idx + dist) % len(cycle)]
    
    return result
