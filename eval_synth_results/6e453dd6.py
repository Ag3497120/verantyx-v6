def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 6
    
    five_col = None
    for c in range(cols):
        if all(grid[r][c] == 5 for r in range(rows)):
            five_col = c
            break
    
    out = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        out[r][five_col] = 5
    
    shape_groups = []
    current = []
    for r in range(rows):
        has_zero = any(grid[r][c] == 0 for c in range(five_col))
        if has_zero:
            current.append(r)
        else:
            if current:
                shape_groups.append(current[:])
            current = []
    if current:
        shape_groups.append(current[:])
    
    for shape in shape_groups:
        max_col = 0
        for r in shape:
            for c in range(five_col):
                if grid[r][c] == 0:
                    max_col = max(max_col, c)
        
        shift = (five_col - 1) - max_col
        
        for r in shape:
            zero_cols_shifted = []
            for c in range(five_col):
                if grid[r][c] == 0:
                    new_c = c + shift
                    if 0 <= new_c < five_col:
                        out[r][new_c] = 0
                        zero_cols_shifted.append(new_c)
            
            if len(zero_cols_shifted) >= 2:
                min_z = min(zero_cols_shifted)
                max_z = max(zero_cols_shifted)
                if max_z == five_col - 1:
                    has_gap = any(out[r][c] != 0 for c in range(min_z, max_z + 1))
                    if has_gap:
                        for c in range(five_col + 1, cols):
                            out[r][c] = 2
    
    return out
