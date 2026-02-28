def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    result = np.zeros((5, 5), dtype=int)
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 6:
                if j + 1 < w and grid[i, j + 1] != 6:
                    patch = grid[i:i+5, j+1:j+6]
                    if patch.shape == (5, 5):
                        result = patch.copy()
                        break
                if i + 1 < h and grid[i + 1, j] != 6:
                    patch = grid[i+1:i+6, j:j+5]
                    if patch.shape == (5, 5):
                        result = patch.copy()
                        break
    return result.tolist()