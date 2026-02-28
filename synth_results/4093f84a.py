def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    
    # Find the solid block of 5s
    five_rows = set()
    five_cols = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_rows.add(r)
                five_cols.add(c)
    
    if not five_rows:
        return [list(row) for row in grid]
    
    r1, r2 = min(five_rows), max(five_rows)
    c1, c2 = min(five_cols), max(five_cols)
    
    # Copy 5-block as-is
    for r in range(r1, r2+1):
        for c in range(c1, c2+1):
            result[r][c] = 5
    
    def is_marker(v):
        return v != 0 and v != 5
    
    # For each row in block range: count markers to left/right
    for r in range(r1, r2+1):
        left_count = sum(1 for c in range(0, c1) if is_marker(grid[r][c]))
        right_count = sum(1 for c in range(c2+1, cols) if is_marker(grid[r][c]))
        for i in range(left_count):
            c = c1 - 1 - i
            if 0 <= c < cols:
                result[r][c] = 5
        for i in range(right_count):
            c = c2 + 1 + i
            if 0 <= c < cols:
                result[r][c] = 5
    
    # For each col in block range: count markers above/below
    for c in range(c1, c2+1):
        top_count = sum(1 for r in range(0, r1) if is_marker(grid[r][c]))
        bot_count = sum(1 for r in range(r2+1, rows) if is_marker(grid[r][c]))
        for i in range(top_count):
            r = r1 - 1 - i
            if 0 <= r < rows:
                result[r][c] = 5
        for i in range(bot_count):
            r = r2 + 1 + i
            if 0 <= r < rows:
                result[r][c] = 5
    
    return result
