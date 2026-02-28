def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                continue
            color = grid[y, x]
            if y + 1 < h and grid[y + 1, x] == 0:
                for dy in range(y + 1, h):
                    if grid[dy, x] != 0:
                        break
                    out[dy, x] = color
                break
    return out.tolist()