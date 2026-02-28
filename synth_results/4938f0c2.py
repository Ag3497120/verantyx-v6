def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    cells_2 = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 2]
    cells_3 = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == 3]
    
    if not cells_2 or not cells_3:
        return grid
    
    # Center of the 3-shape
    r3 = [r for r,c in cells_3]
    c3 = [c for r,c in cells_3]
    cr = (min(r3) + max(r3)) / 2
    cc = (min(c3) + max(c3)) / 2
    
    result = np.zeros_like(g)
    # Place 3-cells
    for r,c in cells_3:
        result[r,c] = 3
    
    # 4 reflections of 2-cells
    for r,c in cells_2:
        for nr, nc in [(r,c), (round(2*cr-r), c), (r, round(2*cc-c)), (round(2*cr-r), round(2*cc-c))]:
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr,nc] = 2
    
    return result.tolist()
