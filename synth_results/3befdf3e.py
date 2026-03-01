def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find bounding box of non-zero region
    r_min = min(r for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    r_max = max(r for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    c_min = min(c for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    c_max = max(c for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    
    border_color = grid[r_min][c_min]  # corner = border
    # inner color: find a cell inside (not on boundary of box)
    inner_color = None
    for r in range(r_min+1, r_max):
        for c in range(c_min+1, c_max):
            if grid[r][c] != border_color:
                inner_color = grid[r][c]
                break
        if inner_color is not None:
            break
    
    if inner_color is None:
        return grid
    
    # inner region size
    inner_h = r_max - r_min - 1  # rows inside border
    inner_w = c_max - c_min - 1
    
    # Build output
    result = [row[:] for row in grid]
    
    # Swap colors inside box: where it was border_color -> inner_color, vice versa
    for r in range(r_min, r_max+1):
        for c in range(c_min, c_max+1):
            if grid[r][c] == border_color:
                result[r][c] = inner_color
            elif grid[r][c] == inner_color:
                result[r][c] = border_color
    
    # Add extension strips of border_color (original outer) around the box
    # Extension size = inner dimensions
    ext_h = inner_h  # extend by inner_h rows top and bottom
    ext_w = inner_w  # extend by inner_w cols left and right
    
    # Top strips
    for dr in range(1, ext_h+1):
        r = r_min - dr
        if r >= 0:
            for c in range(c_min, c_max+1):
                result[r][c] = border_color
    
    # Bottom strips
    for dr in range(1, ext_h+1):
        r = r_max + dr
        if r < rows:
            for c in range(c_min, c_max+1):
                result[r][c] = border_color
    
    # Left strips
    for dc in range(1, ext_w+1):
        c = c_min - dc
        if c >= 0:
            for r in range(r_min, r_max+1):
                result[r][c] = border_color
    
    # Right strips
    for dc in range(1, ext_w+1):
        c = c_max + dc
        if c < cols:
            for r in range(r_min, r_max+1):
                result[r][c] = border_color
    
    return result
