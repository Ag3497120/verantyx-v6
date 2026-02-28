def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    
    # Count 2x2 blocks of 1s
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if g[r,c] == 1 and g[r,c+1] == 1 and g[r+1,c] == 1 and g[r+1,c+1] == 1:
                count += 1
    
    # Output is 1x5 with count 1s followed by 0s
    return [[1] * count + [0] * (5 - count)]
