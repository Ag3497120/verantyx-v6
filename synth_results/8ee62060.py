def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    # Find the block size: the period of the repeating unit
    # Check if block size 2 works (most common)
    for bs in [2, 3, 1, 4]:
        if H % bs == 0:
            blocked = g.reshape(H // bs, bs, W)
            result = np.flipud(blocked).reshape(H, W)
            return result.tolist()
    return grid
