def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.zeros((4, 4), dtype=int)
    
    for i in range(4):
        for j in range(4):
            block = grid[i*4:(i+1)*4, j*4:(j+1)*4]
            if block.shape != (4, 4):
                block = grid[i*4:, j*4:(j+1)*4]
            if block.shape != (4, 4):
                block = grid[i*4:(i+1)*4, j*4:]
            if block.shape != (4, 4):
                block = grid[i*4:, j*4:]
            
            if np.all(block == 0):
                out[i, j] = 0
            else:
                nonzeros = block[block != 0]
                if len(nonzeros) == 0:
                    out[i, j] = 0
                else:
                    counts = np.bincount(nonzeros)
                    max_count = np.max(counts)
                    candidates = np.where(counts == max_count)[0]
                    if len(candidates) == 1:
                        out[i, j] = candidates[0]
                    else:
                        out[i, j] = candidates[-1]
    
    return out.tolist()