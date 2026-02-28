import numpy as np

def transform(grid):
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    output = grid.copy()
    
    for y in range(h):
        for x in range(w):
            val = grid[y, x]
            if val == 2:
                # Right neighbor
                if x + 1 < w and grid[y, x + 1] == 7:
                    output[y, x + 1] = 3
                # Bottom neighbor
                if y + 1 < h and grid[y + 1, x] == 7:
                    output[y + 1, x] = 3
            elif val == 5:
                # Right neighbor
                if x + 1 < w and grid[y, x + 1] == 7:
                    output[y, x + 1] = 4
                # Bottom neighbor
                if y + 1 < h and grid[y + 1, x] == 7:
                    output[y + 1, x] = 4
    return output.tolist()