def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    two_cells = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 2]
    five_cells = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 5]
    
    if not two_cells or not five_cells:
        return grid
    
    max_2r = max(r for r,c in two_cells)
    min_2r = min(r for r,c in two_cells)
    max_2c = max(c for r,c in two_cells)
    min_2c = min(c for r,c in two_cells)
    max_5r = max(r for r,c in five_cells)
    min_5r = min(r for r,c in five_cells)
    max_5c = max(c for r,c in five_cells)
    min_5c = min(c for r,c in five_cells)
    
    # Determine movement direction and offset
    result = np.full_like(g, 7)
    
    if max_2c < min_5c:  # 2 is left of 5, move right
        dr, dc = 0, min_5c - max_2c - 1
        axis = max_2c + dc + 0.5
        new_twos = [(r, c+dc) for r,c in two_cells]
        new_fives = [(r, round(2*axis - c)) for r,c in new_twos]
    elif min_2c > max_5c:  # 2 is right of 5, move left
        dr, dc = 0, max_5c - min_2c + 1
        axis = min_2c + dc - 0.5
        new_twos = [(r, c+dc) for r,c in two_cells]
        new_fives = [(r, round(2*axis - c)) for r,c in new_twos]
    elif max_2r < min_5r:  # 2 is above 5, move down
        dr, dc = min_5r - max_2r - 1, 0
        axis = max_2r + dr + 0.5
        new_twos = [(r+dr, c) for r,c in two_cells]
        new_fives = [(round(2*axis - r), c) for r,c in new_twos]
    else:  # 2 is below 5, move up
        dr, dc = max_5r - min_2r + 1, 0
        axis = min_2r + dr - 0.5
        new_twos = [(r+dr, c) for r,c in two_cells]
        new_fives = [(round(2*axis - r), c) for r,c in new_twos]
    
    for r,c in new_twos:
        if 0 <= r < rows and 0 <= c < cols:
            result[r,c] = 2
    for r,c in new_fives:
        if 0 <= r < rows and 0 <= c < cols:
            result[r,c] = 5
    
    return result.tolist()
