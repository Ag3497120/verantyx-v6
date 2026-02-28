def transform(grid):
    rows, cols = len(grid), len(grid[0])
    g = [list(row) for row in grid]
    
    # find background (most common)
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # find pipe column (column with multiple 8s)
    pipe_col = None
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if col_vals.count(8) >= 2:
            pipe_col = c
            break
    
    if pipe_col is None:
        return grid
    
    # find pipe start row (first row with 8 in pipe_col)
    pipe_start = None
    for r in range(rows):
        if grid[r][pipe_col] == 8:
            pipe_start = r
            break
    
    # find palette row (row with most non-bg non-8 values)
    palette_row = None
    for r in range(rows):
        row = grid[r]
        non_bg = sum(1 for v in row if v != bg and v != 8)
        if non_bg > 0:
            palette_row = r
            break
    
    if palette_row is None or pipe_start is None:
        return grid
    
    palette = grid[palette_row]
    
    # get left values from pipe_col going left (excluding bg)
    left_vals = []
    for c in range(pipe_col-1, -1, -1):
        v = palette[c]
        left_vals.append(v)  # includes bg
    
    right_vals = []
    for c in range(pipe_col+1, cols):
        v = palette[c]
        right_vals.append(v)  # includes bg
    
    # Build output: start with all bg
    out = [[bg]*cols for _ in range(rows)]
    
    # Place pipe of 8s
    for r in range(rows):
        if grid[r][pipe_col] == 8:
            out[r][pipe_col] = 8
    
    # At each row from pipe_start downward, radiate palette values
    max_extent = max(len(left_vals), len(right_vals))
    
    for r in range(pipe_start, rows):
        if out[r][pipe_col] != 8:
            break
        dist = r - pipe_start  # 0 at pipe_start
        
        # place left values
        for i in range(dist):
            tc = pipe_col - 1 - i
            if 0 <= tc < cols and i < len(left_vals):
                out[r][tc] = left_vals[i]
        
        # place right values
        for i in range(dist):
            tc = pipe_col + 1 + i
            if 0 <= tc < cols and i < len(right_vals):
                out[r][tc] = right_vals[i]
    
    return out
