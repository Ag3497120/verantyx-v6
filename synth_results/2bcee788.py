def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    twos = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 2]
    xs = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] != 0 and g[r,c] != 2]
    
    if not twos or not xs:
        return grid
    
    x_color = g[xs[0][0], xs[0][1]]
    
    max_x_col = max(c for r,c in xs)
    min_x_col = min(c for r,c in xs)
    max_2_col = max(c for r,c in twos)
    min_2_col = min(c for r,c in twos)
    max_x_row = max(r for r,c in xs)
    min_x_row = min(r for r,c in xs)
    max_2_row = max(r for r,c in twos)
    min_2_row = min(r for r,c in twos)
    
    if max_x_col < min_2_col:
        axis = (max_x_col + min_2_col) / 2
        reflected = [(r, round(2*axis - c)) for r,c in xs]
    elif min_x_col > max_2_col:
        axis = (max_2_col + min_x_col) / 2
        reflected = [(r, round(2*axis - c)) for r,c in xs]
    elif max_x_row < min_2_row:
        axis = (max_x_row + min_2_row) / 2
        reflected = [(round(2*axis - r), c) for r,c in xs]
    else:
        axis = (max_2_row + min_x_row) / 2
        reflected = [(round(2*axis - r), c) for r,c in xs]
    
    result = np.full_like(g, 3)
    for r,c in xs:
        result[r,c] = x_color
    for r,c in reflected:
        if 0 <= r < rows and 0 <= c < cols:
            result[r,c] = x_color
    
    return result.tolist()
