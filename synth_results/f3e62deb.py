def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    # Find non-background cells
    nz = [(r,c,grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    if not nz: return grid
    
    # Get shape color (only one color)
    color = nz[0][2]
    min_r = min(r for r,c,v in nz)
    max_r = max(r for r,c,v in nz)
    min_c = min(c for r,c,v in nz)
    max_c = max(c for r,c,v in nz)
    
    # Color determines direction
    # 6 -> UP, 8 -> RIGHT, 4 -> DOWN, others -> LEFT (guess)
    color_dir = {6: 'UP', 8: 'RIGHT', 4: 'DOWN', 3: 'LEFT', 1: 'LEFT', 2: 'LEFT', 7: 'LEFT'}
    direction = color_dir.get(color, 'UP')
    
    # Move shape
    result = [[0]*cols for _ in range(rows)]
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1
    
    if direction == 'UP':
        # Move to top row
        dr = -min_r
        dc = 0
    elif direction == 'DOWN':
        # Move to bottom row
        dr = (rows-1) - max_r
        dc = 0
    elif direction == 'RIGHT':
        # Move to right column
        dr = 0
        dc = (cols-1) - max_c
    elif direction == 'LEFT':
        # Move to left column
        dr = 0
        dc = -min_c
    
    for r,c,v in nz:
        new_r = r + dr
        new_c = c + dc
        if 0 <= new_r < rows and 0 <= new_c < cols:
            result[new_r][new_c] = v
    
    return result
