def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    for i in range(h):
        for j in range(w):
            val = grid[i, j]
            if val == 4:
                continue
            if i > 0 and grid[i-1, j] == 4:
                out[i, j] = 3
            elif i < h-1 and grid[i+1, j] == 4:
                out[i, j] = 3
            elif j > 0 and grid[i, j-1] == 4:
                out[i, j] = 3
            elif j < w-1 and grid[i, j+1] == 4:
                out[i, j] = 3
            elif i > 0 and j > 0 and grid[i-1, j-1] == 4:
                out[i, j] = 2
            elif i > 0 and j < w-1 and grid[i-1, j+1] == 4:
                out[i, j] = 2
            elif i < h-1 and j > 0 and grid[i+1, j-1] == 4:
                out[i, j] = 2
            elif i < h-1 and j < w-1 and grid[i+1, j+1] == 4:
                out[i, j] = 2
    return out.tolist()